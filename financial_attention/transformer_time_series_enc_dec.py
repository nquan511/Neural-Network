"""
Architecture Overview:
--------------------
- Encoder-Decoder Transformer with ProbSparse Self-Attention
- Multi-head attention mechanism for capturing complex temporal dependencies
- Distilling mechanism in encoder for handling long sequences efficiently
- Causal masking in decoder to prevent future information leakage

Key Components:
-------------
1. Attention Mechanisms:
   - Standard Multi-head Attention
   - ProbSparse Self-Attention for efficient computation
   - Cross-attention for encoder-decoder interaction

2. Model Structure:
   - Encoder: Processes historical data with distillation
   - Decoder: Generates predictions with causal attention
   - Positional Encoding: Provides temporal information

3. Training Features:
   - Learning rate scheduling with warmup and decay
   - Early stopping with best model restoration
   - Mixed precision training support
   - Memory-efficient attention computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np
from dataclasses import dataclass
import time
import inspect
import logging
import plotly.express as px

# ---------------------------------------------------------
# Logging configuration (production-friendly)
# ---------------------------------------------------------
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger("informer_forecaster")

def generate_causal_mask(seq_len, device):
    """
    Creates a lower-triangular mask of shape (seq_len, seq_len),
    where True means 'keep' (visible), and False means 'mask out'.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask


# =====================================================
# Attention Mechanisms
# =====================================================

class Attention(nn.Module):
    """
    Multi-head Attention mechanism with separate attention heads for parallel processing.
    
    This implementation follows the original "Attention Is All You Need" paper with
    additional optimizations for stability and regularization.
    
    Architecture:
    ------------
    1. Input Projections:
       - Query (Q): Linear projection of query input
       - Key (K): Linear projection of key input
       - Value (V): Linear projection of value input
       
    2. Attention Computation:
       - Scaled dot-product attention: (Q @ K.T) / sqrt(d_k)
       - Optional masking for causal attention
       - Softmax normalization
       - Dropout for regularization
       
    3. Output Processing:
       - Multi-head concatenation
       - Final linear projection
       - Output dropout
    
    Args:
        d_model (int): Model dimension, must be divisible by n_heads
        n_heads (int): Number of attention heads
        attention_dropout (float): Dropout probability for attention and output
        mask_flag (bool): Whether to support attention masking
        
    Shape:
        - Input: (batch_size, seq_length, d_model)
        - Output: (batch_size, seq_length, d_model)
        - Attention mask: (seq_length, seq_length) or (batch_size, n_heads, seq_length, seq_length)
    """
    def __init__(self, d_model, n_heads, attention_dropout=0.1, mask_flag=True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.mask_flag = mask_flag

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        # Apply dropout to both attention weights and output
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(attention_dropout)

    def forward(self, Q, K, V, attn_mask=None):
        # Extract input dimensions
        # B: batch size, L_Q: query sequence length, D: model dimension
        B, L_Q, D = Q.shape
        # L_K: key sequence length (same as value sequence length)
        _, L_K, _ = K.shape

        # Multi-head projection and reshaping:
        # 1. Linear projection to d_model dimensions
        # 2. Reshape to separate heads: [B, L, D] -> [B, L, H, D/H]
        # 3. Transpose for attention: [B, L, H, D/H] -> [B, H, L, D/H]
        # Final shape: [batch_size, n_heads, seq_length, d_k]
        Q = self.q_proj(Q).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(K).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(V).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        # 1. Matrix multiply Q and K^T: [B, H, L_Q, d_k] @ [B, H, d_k, L_K]
        # 2. Scale by sqrt(d_k) for stable gradients
        # Result shape: [batch_size, n_heads, L_Q, L_K]
        # Compute attention scores with scaling
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply attention masking if provided
        if attn_mask is not None:
            # Handle different mask shapes:
            # - 2D mask [L_Q, L_K]: Expand to 4D [1, 1, L_Q, L_K]
            # - 4D mask [B, H, L_Q, L_K]: Use as is
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            # Replace masked positions with -inf before softmax
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Compute attention weights:
        # 1. Apply softmax to get probabilities
        # 2. Apply dropout for regularization
        # Shape: [batch_size, n_heads, L_Q, L_K]
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Compute output:
        # 1. Matrix multiply with values: [B, H, L_Q, L_K] @ [B, H, L_K, d_k]
        # 2. Transpose and reshape back to original dimensions
        # 3. Project to final output space
        # 4. Apply output dropout
        out = torch.matmul(attn, V)  # [B, H, L_Q, d_k]
        out = out.transpose(1, 2).contiguous().view(B, L_Q, D)  # [B, L_Q, D]
        out = self.o_proj(out)  # Final projection
        return self.out_dropout(out)

class ProbSparseSelfAttention(Attention):
    """
    ProbSparse Self-Attention mechanism that improves efficiency by selecting dominant queries.
    
    This implementation is based on the Informer paper's ProbSparse attention mechanism,
    which reduces the quadratic complexity of vanilla attention while maintaining performance.
    
    Key Features:
    ------------
    1. Query Sparsification:
       - Samples a subset of keys for initial scoring
       - Identifies top queries that are most likely to contribute to the output
       - Uses these dominant queries for full attention computation
       
    2. Memory Efficiency:
       - Reduces memory usage from O(L_Q * L_K) to O(u * L_K), where u << L_Q
       - Maintains attention quality through smart query selection
       - Uses no_grad for mean calculations to save memory
       
    3. Numerical Stability:
       - Adds epsilon to prevent underflow
       - Uses logsumexp for stable probability computation
       - Properly handles attention masking
    
    Args:
        d_model (int): Model dimension, must be divisible by n_heads
        n_heads (int): Number of attention heads
        attention_dropout (float): Dropout probability
        mask_flag (bool): Whether to support attention masking
        factor (int): Factor for computing number of top queries (u = factor * log(L_K))
        
    Implementation Notes:
    -------------------
    - The sparsification happens in query space rather than key space
    - Uses a sampling-based approach to estimate query importance
    - Automatically adapts to sequence length through logarithmic scaling
    """
    def __init__(self, d_model, n_heads, attention_dropout=0.1, mask_flag=True, factor=5):
        super().__init__(d_model, n_heads, attention_dropout, mask_flag)
        self.factor = factor

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Core function of ProbSparse attention that identifies the most important query positions.
        Uses a sampling-based approach to approximate query sparsity scores.
        """
        # Extract dimensions
        # B: batch size, H: num heads, L_Q: query length, D: head dimension
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Sample a subset of keys for efficient scoring
        # k: number of keys to sample (min of sample_k or available keys)
        k = min(sample_k, L_K)
        # Randomly select k indices from the key sequence
        index_sample = torch.randint(L_K, (k,), device=K.device)
        # Extract the sampled keys
        K_sample = K[:, :, index_sample, :]

        # Compute attention scores for sampled keys
        # Shape: [B, H, L_Q, k]
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / math.sqrt(D)
        eps = 1e-8  # Prevent log(0) and numerical underflow

        # Compute mean attention score (no gradients needed)
        # This represents the average interaction strength
        with torch.no_grad():
            mean_K = Q_K_sample.mean(dim=-1, keepdim=True)

        # Compute sparsity scores:
        # 1. Add epsilon for numerical stability
        # 2. Use logsumexp for stable probability computation
        # 3. Subtract mean to get relative importance
        M = torch.logsumexp(Q_K_sample + eps, dim=-1) - mean_K.squeeze(-1)

        # Select top-n query positions based on sparsity scores
        # Returns indices of top n_top queries for each batch and head
        M_top = torch.topk(M, n_top, dim=-1)[1]
        return M_top

    def forward(self, Q, K, V, attn_mask=None):
        """
        Forward pass of ProbSparse attention with efficient sparse computation.
        Only computes full attention for the most important query positions.
        """
        # Project and reshape inputs to multi-head format
        B, L_Q, D = Q.shape
        _, L_K, _ = K.shape
        Q = self.q_proj(Q).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(K).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(V).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        # Calculate sampling parameters based on sequence lengths:
        # - sample_k: number of keys to sample for scoring queries
        # - n_top: number of top queries to compute full attention for
        factor = self.factor
        sample_k = max(1, int(factor * math.log(max(L_K, 2))))
        n_top = min(L_Q, max(1, int(factor * math.log(max(L_Q, 2)))))

        # Identify the most important query positions
        M_top = self._prob_QK(Q, K, sample_k, n_top)

        # Initialize output context with mean values (no gradients needed)
        with torch.no_grad():
            context = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1).clone()

        # Extract top queries and compute their attention scores
        # Shape: [B, H, n_top, d_k]
        Q_top = torch.gather(Q, 2, M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k))
        attn_scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Handle attention masking for top queries
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            # Gather mask values for top queries
            attn_mask_subset = torch.gather(attn_mask.expand(B, self.n_heads, L_Q, L_K), 2, 
                                          M_top.unsqueeze(-1).expand(-1, -1, -1, L_K))
            attn_scores = attn_scores.masked_fill(attn_mask_subset == 0, float('-inf'))

        # Compute attention weights and apply to values
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        # Compute context vectors for top queries
        context_top = torch.matmul(attn, V)

        # Update context with computed values at top query positions
        # Uses scatter_ to place computed attention results in the right positions
        context.scatter_(2, M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k), context_top)

        # Reshape to output format and apply final projection
        out = context.transpose(1, 2).contiguous().view(B, L_Q, D)
        return self.o_proj(out)

# =====================================================
# Encoder/Decoder Architecture
# =====================================================

class EncoderLayer(nn.Module):
    """
    Single layer of the encoder stack with self-attention and optional distillation.
    
    This layer implements the pre-norm transformer architecture with an additional
    distillation mechanism for processing long sequences efficiently.
    
    Architecture:
    ------------
    1. Self-Attention Block:
       - ProbSparse self-attention for efficient processing
       - Pre-normalization for training stability
       - Residual connection and dropout
       
    2. Feed-Forward Block:
       - Two linear transformations with GELU activation
       - Pre-normalization and residual connection
       - Dropout for regularization
       
    3. Optional Distillation (if enabled):
       - Convolutional layer for local feature extraction
       - ELU activation for smooth gradients
       - MaxPool for sequence length reduction
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward layer dimension
        dropout (float): Dropout probability
        distill (bool): Whether to use distillation
        factor (int): Factor for ProbSparse attention
    
    Shape:
        - Input: (batch_size, seq_length, d_model)
        - Output: (batch_size, seq_length/2 if distill else seq_length, d_model)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, distill=True, factor=5):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads, factor=factor)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.distill = distill
        if distill:
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.ELU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )

    def forward(self, x):
        x = x + self.dropout(self.attn(x, x, x))
        x = self.norm1(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm2(x)
        if self.distill:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x

class DecoderLayer(nn.Module):
    """
    Single layer of the decoder stack with self-attention and cross-attention mechanisms.
    
    This layer implements a pre-norm transformer decoder architecture with both
    masked self-attention and cross-attention to the encoder outputs.
    
    Architecture:
    ------------
    1. Masked Self-Attention:
       - ProbSparse self-attention with causal masking
       - Pre-normalization for stability
       - Residual connection and dropout
       
    2. Cross-Attention:
       - Standard attention to encoder outputs
       - Enables information flow from encoder
       - Pre-normalization and residual connection
       
    3. Feed-Forward:
       - Two linear transformations with GELU
       - Final pre-normalization
       - Residual connection and dropout
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward layer dimension
        dropout (float): Dropout probability
        factor (int): Factor for ProbSparse attention
    
    Shape:
        - Decoder input: (batch_size, target_length, d_model)
        - Encoder input: (batch_size, source_length, d_model)
        - Output: (batch_size, target_length, d_model)
        - Mask: (target_length, target_length)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, factor=5):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, factor=factor)
        self.cross_attn = Attention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec, enc,attn_mask=None):
        dec = dec + self.dropout(self.self_attn(dec, dec, dec, attn_mask=attn_mask))
        dec = self.norm1(dec)
        dec = dec + self.dropout(self.cross_attn(dec, enc, enc))
        dec = self.norm2(dec)
        dec = dec + self.dropout(self.ff(dec))
        dec = self.norm3(dec)
        return dec

class Encoder(nn.Module):
    """
    Complete encoder stack consisting of multiple EncoderLayers with distillation.
    
    The encoder processes the input sequence through multiple layers of self-attention
    and feed-forward networks, with optional progressive distillation for handling
    long sequences efficiently.
    
    Architecture:
    ------------
    1. Layer Structure:
       - Multiple EncoderLayers in sequence
       - Progressive distillation (except last layer if enabled)
       - Consistent dimensionality throughout
       
    2. Sequence Processing:
       - Each layer processes entire sequence
       - Distillation reduces sequence length progressively
       - Final layer always maintains full dimensionality
       
    3. Feature Extraction:
       - Hierarchical feature learning
       - Increasingly global context through depth
       - Memory-efficient processing of long sequences
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward layer dimension
        n_layers (int): Number of encoder layers
        dropout (float): Dropout probability
        distill (bool): Whether to use distillation
        factor (int): Factor for ProbSparse attention
    
    Shape:
        - Input: (batch_size, seq_length, d_model)
        - Output: (batch_size, seq_length/(2^(n_layers-1)) if distill else seq_length, d_model)
    """
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1, distill=True, factor=5):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, distill=(distill and i < (n_layers - 1)), factor=factor)
            for i in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    """
    Complete decoder stack consisting of multiple DecoderLayers with causal attention.
    
    The decoder generates the output sequence autoregressively, using both self-attention
    with causal masking and cross-attention to the encoder outputs.
    
    Architecture:
    ------------
    1. Layer Structure:
       - Multiple DecoderLayers in sequence
       - Each layer has masked self-attention and cross-attention
       - Consistent dimensionality throughout
       
    2. Sequence Generation:
       - Causal masking prevents looking at future tokens
       - Cross-attention to full encoder context
       - Progressive refinement through layers
       
    3. Information Flow:
       - Self-attention for target sequence coherence
       - Cross-attention for source sequence conditioning
       - Deep processing through multiple layers
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward layer dimension
        n_layers (int): Number of decoder layers
        dropout (float): Dropout probability
        factor (int): Factor for ProbSparse attention
    
    Shape:
        - Decoder input: (batch_size, target_length, d_model)
        - Encoder input: (batch_size, source_length, d_model)
        - Output: (batch_size, target_length, d_model)
    """
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1, factor=5):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(n_layers)
        ])

    def forward(self, dec, enc):
        seq_len = dec.size(1)
        device = dec.device
        attn_mask = generate_causal_mask(seq_len, device)
        for layer in self.layers:
            dec = layer(dec, enc, attn_mask=attn_mask)
        return dec

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
            
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum length {self.max_len}")
        return x + self.pe[:, :seq_len]

    @classmethod
    def create_with_auto_max_len(cls, d_model, enc_len, pred_len, safety_factor=1.5):
        """Factory method to create instance with automatically calculated max_len"""
        max_len = int((enc_len + pred_len) * safety_factor)
        return cls(d_model, max_len)


# Informer Forecaster (unchanged)
class TimeEmbedding(nn.Module):
    """
    Embeddings for different time granularities (minute, hour, day)
    """
    def __init__(self, d_model, max_period, embedding_type="hour"):
        super().__init__()
        self.embedding = nn.Embedding(max_period, d_model)
        self.embedding_type = embedding_type

    def forward(self, timestamps):
        """
        Args:
            timestamps: Tensor of shape [batch_size, seq_len, 3] where the last dimension contains
                       [minute, hour, day] components in that order
        """
        if self.embedding_type == "hour":
            time_component = timestamps[:, :, 0]  
        elif self.embedding_type == "day":
            time_component = timestamps[:, :, 1] 
        elif self.embedding_type == "month":
            time_component = timestamps[:, :, 2]  
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        
        return self.embedding(time_component)

class InformerForecaster(nn.Module):
    def __init__(self, config, asset_index=0):
        super().__init__()
        self.asset_index = asset_index
        self.d_input = config["d_input"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_ff = config["d_ff"]
        self.enc_layers = config["enc_layers"]
        self.dec_layers = config["dec_layers"]
        self.dropout = config["dropout"]
        self.distill = config["distill"]
        self.factor = config["factor"]

        # Input embeddings
        self.enc_embedding = nn.Linear(self.d_input, self.d_model)
        self.dec_embedding = nn.Linear(self.d_input, self.d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(self.d_model)
        
        # Time embeddings for different granularities
        self.month_embedding = TimeEmbedding(self.d_model, max_period=12, embedding_type="month")
        self.hour_embedding = TimeEmbedding(self.d_model, max_period=24, embedding_type="hour")
        self.day_embedding = TimeEmbedding(self.d_model, max_period=31, embedding_type="day")

        self.encoder = Encoder(self.d_model, self.n_heads, self.d_ff, self.enc_layers,
                               dropout=self.dropout, distill=self.distill, factor=self.factor)
        self.decoder = Decoder(self.d_model, self.n_heads, self.d_ff, self.dec_layers,
                               dropout=self.dropout, factor=self.factor)
        self.proj = nn.Linear(self.d_model, 1)

        self.enc_len = config["enc_len"]
        self.guiding_len = config["guiding_len"]
        self.pred_len = config["pred_len"]

    def forward(self, seq, timestamps=None):
        """
        seq: [B, L_total, d_input]
        Must have L_total >= enc_len + pred_len
        timestamps: [B, L_total] optional, datetime tensor for time embeddings
        """
        if seq.size(1) < self.enc_len + self.pred_len:
            raise ValueError(f"Input sequence length {seq.size(1)} is too short. "
                           f"Required length: {self.enc_len + self.pred_len}")
        
        device = seq.device
        B = seq.size(0)

        # Encoder processing
        enc_x = seq[:, :self.enc_len, :]
        enc_h = self.pos_enc(self.enc_embedding(enc_x))
        enc_out = self.encoder(enc_h)

        # Decoder input = guiding context + zeros
        dec_context = seq[:, self.enc_len - self.guiding_len: self.enc_len, :]
        dec_future = seq[:, self.enc_len: self.enc_len + self.pred_len, :].clone()
        # Use proper device for zero tensor
        dec_future[:, :, self.asset_index] = torch.zeros(dec_future.size(0), dec_future.size(1), 
                                                        device=device)
        dec_input = torch.cat([dec_context, dec_future], dim=1)
        dec_h = self.pos_enc(self.dec_embedding(dec_input))
        
        # Add time embeddings to decoder if provided
        if timestamps is not None:
            dec_timestamps = timestamps[:, self.enc_len - self.guiding_len: self.enc_len + self.pred_len]
            month_emb = self.month_embedding(dec_timestamps)
            hour_emb = self.hour_embedding(dec_timestamps)
            day_emb = self.day_embedding(dec_timestamps)
            dec_h = dec_h + month_emb + hour_emb + day_emb
            
        dec_out = self.decoder(dec_h, enc_out)

        # Only predict on placeholder region
        out = self.proj(dec_out[:, -self.pred_len:, :])
        return out.squeeze(-1)

# =====================================================
# Dataset and Data Loading Utilities
# =====================================================

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time series data with sliding window sampling and timestamps.
    
    This dataset creates training/validation samples by sliding a window over
    the input time series, with each sample containing both input (encoder)
    and target (prediction) sequences along with their timestamps.
    
    Features:
    --------
    1. Sliding Window:
       - Window size = enc_len + pred_len
       - Stride = 1 (maximum overlap)
       - Automatic sample count calculation
       
    2. Time Features:
       - Preserves datetime information
       - Supports minute, hour, day embeddings
       - Maintains temporal ordering
       
    Args:
        data (np.ndarray): Input time series data
        timestamps (pd.DatetimeIndex): Corresponding timestamps for each data point
        enc_len (int): Length of input sequence for encoder
        pred_len (int): Length of target sequence to predict
        
    Shape:
        - Input data: (total_length, feature_dim)
        - Each sample: (enc_len + pred_len, feature_dim)
        - Timestamps: (enc_len + pred_len,)
    """
    def __init__(self, data, timestamps, enc_len, pred_len):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        if enc_len <= 0 or pred_len <= 0:
            raise ValueError("enc_len and pred_len must be positive integers")
        if len(data) < enc_len + pred_len:
            raise ValueError(f"Data length {len(data)} is too short for enc_len={enc_len} and pred_len={pred_len}")
        if len(data) != len(timestamps):
            raise ValueError(f"Data length {len(data)} must match timestamps length {len(timestamps)}")
        
        self.data = data
        self.timestamps = timestamps
        self.enc_len = enc_len
        self.pred_len = pred_len
        self.samples = len(data) - (enc_len + pred_len) + 1
        
    def __len__(self):
        return self.samples

    def _convert_timestamps_to_tensor(self, timestamps):
        """Convert timestamps to tensors with minute, hour, and day components"""
        hours = torch.tensor(timestamps.hour.values, dtype=torch.long)
        weekdays = torch.tensor(timestamps.weekday.values, dtype=torch.long)
        months = torch.tensor(timestamps.month.values - 1, dtype=torch.long)
        
        # Stack time components into a tensor [seq_len, 3]
        time_tensor = torch.stack([hours, weekdays, months], dim=1)
        return time_tensor

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with {self.samples} samples")
            
        # Get sequence data
        seq = self.data[idx: idx + self.enc_len + self.pred_len]
        if len(seq) != self.enc_len + self.pred_len:
            raise RuntimeError(f"Unexpected sequence length {len(seq)}, expected {self.enc_len + self.pred_len}")
            
        # Get corresponding timestamps
        ts = self.timestamps[idx: idx + self.enc_len + self.pred_len]
        time_tensor = self._convert_timestamps_to_tensor(ts)
        
        return torch.tensor(seq, dtype=torch.float32), time_tensor

def create_dataloaders(df, enc_len=96, pred_len=24, batch_size=32, val_batch_size=1, val_ratio=0.1, asset_name="SOL"):
    """
    Create training and validation dataloaders with proper data preprocessing and timestamp handling.
    
    This function handles the complete data preparation pipeline:
    1. Train/validation splitting
    2. Feature scaling
    3. Timestamp extraction and processing
    4. Dataset creation
    5. DataLoader configuration
    
    Features:
    --------
    1. Data Splitting:
       - Time-based train/validation split
       - Configurable validation ratio
       - No data leakage between sets
       
    2. Preprocessing:
       - StandardScaler for feature normalization
       - Scaler fit on training data only
       - Same scaler applied to validation set
       
    3. Time Features:
       - Extracts timestamps from DataFrame index
       - Preserves temporal information
       - Supports minute, hour, day embeddings
       
    4. DataLoader Configuration:
       - Batch processing with timestamps
       - Shuffling for training
       - Efficient data loading
       
    Args:
        df (pd.DataFrame): Input dataframe with datetime index and time series features
        enc_len (int): Encoder sequence length
        pred_len (int): Prediction sequence length
        batch_size (int): Number of samples per batch
        val_ratio (float): Fraction of data to use for validation
        asset_name (str): Name of the target asset column
        
    Returns:
        tuple:
            - train_loader (DataLoader): Training data loader with timestamps
            - val_loader (DataLoader): Validation data loader with timestamps
            - scaler (StandardScaler): Fitted scaler for inverse transforms
            - asset_index (int): Column index of the target asset
    """
    asset_index = df.columns.get_loc(asset_name)

    # Ensure DataFrame has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    # Split first (no data leakage!)
    n = len(df)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:-n_val]
    val_df = df.iloc[-(n_val + enc_len + pred_len):]

    # Fit scaler only on training set
    scaler = StandardScaler()
    scaler.fit(train_df.values)

    # Transform both sets
    train_scaled = scaler.transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)

    # Create datasets with timestamps
    train_ds = TimeSeriesDataset(train_scaled, train_df.index, enc_len, pred_len)
    val_ds = TimeSeriesDataset(val_scaled, val_df.index, enc_len, pred_len)

    # Custom collate function to handle both data and timestamps
    def collate_fn(batch):
        # Separate sequences and timestamps
        seqs, times = zip(*batch)
        # Stack them into batches
        seqs = torch.stack(seqs)
        times = torch.stack(times)
        return seqs, times

    # Create dataloaders with custom collate function
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, 
                          drop_last=False, collate_fn=collate_fn)

    return train_loader, val_loader, scaler, asset_index

def inverse_transform(y_scaled, scaler, asset_index, n_features):
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, asset_index] = y_scaled
    y_real = scaler.inverse_transform(dummy)[:, asset_index]
    return y_real

# =====================================================
# Optimized Training Utilities
# =====================================================

@dataclass
class TrainConfig:
    """
    Configuration class for model training with comprehensive hyperparameter management.
    
    This class provides a structured way to manage all training-related parameters
    with validation and sensible defaults. It supports advanced training features
    like gradient accumulation, mixed precision, and early stopping.
    
    Features:
    --------
    1. Optimizer Configuration:
       - Learning rate with minimum bound
       - Weight decay for regularization
       - Gradient clipping options
       
    2. Training Process:
       - Maximum training steps
       - Warmup period configuration
       - Early stopping with patience
       
    3. Hardware Optimization:
       - Automatic device selection
       - Mixed precision training support
       
    4. Learning Rate Scheduling:
       - Warmup phase
       - Cosine decay to minimum lr
       - Schedule enable/disable option

    """
    # optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    min_lr: float = 1e-6  # Minimum learning rate
    # training steps
    max_steps: int = 1000
    warmup_steps: int = 100
    # gradients
    grad_clip: float = 1.0
    use_grad_clip: bool = True
    # AMP
    use_amp: bool = True
    # LR schedule
    use_lr_schedule: bool = True
    # Early stopping
    patience: int = 5  # Number of validation checks without improvement before stopping
    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps <= self.warmup_steps:
            raise ValueError("max_steps must be greater than warmup_steps")
        if not 0 <= self.min_lr <= self.learning_rate:
            raise ValueError("min_lr must be between 0 and learning_rate")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")

def _init_weights(module: nn.Module, std: float = 0.02, n_layer: int = 12):
    """
    partial/gpt-style initialization for linear and embedding layers (optional)
    """
    if isinstance(module, nn.Linear):
        local_std = std
        if hasattr(module, "NANOGPT_SCALE_INIT") and getattr(module, "NANOGPT_SCALE_INIT"):
            local_std = std * (2 * n_layer) ** -0.5
        nn.init.normal_(module.weight, mean=0.0, std=local_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        # layernorm defaults are fine
        pass

def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float, device_type: str = "cpu"):
    """
    Configure an optimized AdamW optimizer with weight decay split and fused computation.
    
    This implementation follows best practices for transformer optimization:
    1. Separate weight decay for different parameter types
    2. Fused AdamW operations on CUDA when available
    3. Optimal beta values for transformer training
    
    Features:
    --------
    1. Parameter Grouping:
       - Applies weight decay only to weight matrices
       - Zero weight decay for biases and 1D parameters
       - Automatic parameter classification
       
    2. Hardware Optimization:
       - Automatic fused operation detection
       - CUDA-aware optimization
       - Efficient memory usage
       
    3. Optimizer Configuration:
       - AdamW with transformer-tuned betas
       - Small epsilon for numerical stability
       - Automatic parameter grouping
    
    Args:
        model (nn.Module): The model whose parameters will be optimized
        weight_decay (float): Weight decay factor for regularization
        learning_rate (float): Initial learning rate
        device_type (str): Device type ('cuda' or 'cpu')
        
    Returns:
        torch.optim.AdamW: Configured optimizer
        
    Note:
        Uses fused operations when available on CUDA for better performance
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    log.info(f"Using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

def get_lr(step: int, cfg: TrainConfig):
    """Calculate learning rate with warmup and cosine decay to min_lr"""
    if not cfg.use_lr_schedule:
        return cfg.learning_rate
    
    # Handle warmup phase
    if step < cfg.warmup_steps:
        return cfg.learning_rate * float(step) / max(1, cfg.warmup_steps)
    
    # Calculate progress after warmup
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    
    # Cosine decay from learning_rate to min_lr
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * cosine

# =====================================================
# === Training Loop (AMP, gradclip, LR schedule) ===
# =====================================================
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                cfg: TrainConfig,
                asset_index: int = 0):
    device = torch.device(cfg.device)
    model = model.to(device)

    opt = configure_optimizers(model, weight_decay=cfg.weight_decay, learning_rate=cfg.learning_rate, device_type=cfg.device)
    scaler = torch.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    step = 0
    start_time = time.time()
    train_loss_history = []
    val_loss_history = []
    steps_history = []

    loss_fn = nn.MSELoss()

    model.train()
    while step < cfg.max_steps:
        for batch_idx, (batch_data, batch_timestamps) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_timestamps = batch_timestamps.to(device)
            # true target region: last pred_len timesteps, column asset_index
            y_true = batch_data[:, -model.pred_len:, asset_index]

            # schedule lr
            lr = get_lr(step, cfg)
            for g in opt.param_groups:
                g['lr'] = lr

            opt.zero_grad()
            # autocast context
            autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, 
                                        enabled=(cfg.use_amp and device.type == "cuda"))
            with autocast_ctx:
                y_pred = model(batch_data,batch_timestamps)  # [B, pred_len]
                loss = loss_fn(y_pred, y_true)

            # backward + scaling
            if cfg.use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                if cfg.use_grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

            # logging
            if step % 10 == 0:
                elapsed = time.time() - start_time
                tok_s = (batch_data.shape[0] * batch_data.shape[1]) / max(1e-9, elapsed)
                log.info(f"step {step} | train loss {loss.item():.6f} | lr {lr:.3e} | tok/s {tok_s:.1f}")
                train_loss_history.append(loss.item())
                steps_history.append(step)
                start_time = time.time()

            # validation
            if step % 100 == 0 and val_loader is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_data, val_timestamps in val_loader:
                        val_data = val_data.to(device)
                        val_timestamps = val_timestamps.to(device)
                        vy_true = val_data[:, -model.pred_len:, asset_index]
                        vy_pred = model(val_data,val_timestamps)  # [B, pred_len]
                        val_losses.append(loss_fn(vy_pred, vy_true).item())
                mean_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float('nan')
                val_loss_history.append((step, mean_val_loss))
                log.info(f"step {step} | VALIDATION loss {mean_val_loss:.6f}")
                model.train()

            step += 1
            if step >= cfg.max_steps:
                break

    return model, train_loss_history, val_loss_history, steps_history

# =====================================================
# Example configuration/test run when run as script
# =====================================================
if __name__ == "__main__":
    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    csv_path = r"D:\Quan\Quants\Neural Network\financial_attention\1h_data_20220101_20250601.csv"
    closes = pd.read_csv(csv_path, index_col=0, parse_dates=True)[['SOL', 'ETH', 'BTC','ADA','XRP','LTC','TRX','LINK','DOT','DOGE']]


    config = {
        "d_input": len(closes.columns),
        "d_model": 16,
        "n_heads": 4,
        "d_ff": 16,
        "enc_layers": 3,
        "dec_layers": 2,
        "dropout": 0.0,
        "distill": True,
        "enc_len": 96,
        "guiding_len": 24,
        "pred_len": 1,
        "factor": 5,
    }

    train_loader, val_loader, scaler, asset_idx = create_dataloaders(closes, enc_len=config["enc_len"],
                                                                        pred_len=config["pred_len"],
                                                                        batch_size=32, val_ratio=0.1, asset_name="SOL")
    model = InformerForecaster(config, asset_index=asset_idx)

    # training config
    tcfg = TrainConfig(learning_rate=1e-4, weight_decay=0.01, max_steps=10, warmup_steps=1, use_amp=False, device="cpu")
    model, train_hist, val_hist, steps_hist = train_model(model, train_loader, val_loader, tcfg, asset_index=asset_idx)


    total_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_learnable_params:,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move model to the correct device
    preds = []
    targets = []
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            y_pred = model(x)  # [B, pred_len]
            y_true = x[:, -model.pred_len:, asset_idx]  # [B, pred_len]
            # Append the full predicted sequence, not just last step
            preds.append(y_pred.cpu().numpy())
            targets.append(y_true.cpu().numpy())
    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()

    # Inverse transform to get actual prices
    preds_price = inverse_transform(preds, scaler, asset_idx, config["d_input"])
    targets_price = inverse_transform(targets, scaler, asset_idx, config["d_input"])

    df_plot = pd.DataFrame({
        "Time Step": np.arange(len(targets_price)),
        "Actual Price": targets_price,
        "Predicted Price": preds_price
    })
    df_plot = df_plot.melt(id_vars="Time Step", value_vars=["Actual Price", "Predicted Price"],
                            var_name="Type", value_name="Price")

    fig = px.line(df_plot, x="Time Step", y="Price", color="Type",
                    title="Predicted vs Actual Asset Price (Validation Set)")
    fig.show()