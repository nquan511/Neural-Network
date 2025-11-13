"""
Shared Transformer components for financial time series (Informer-style).

This module centralizes common building blocks used by both the encoder-only and
encoder-decoder models so they can be debugged and evolved in one place:

- Attention blocks (full and ProbSparse)
- Encoder/Decoder layers and stacks with optional distillation
- Positional and time embeddings
- Dataset + dataloaders + inverse transform helpers
- Training utilities (optimizer config, LR schedule, generic training loop)

Design notes
- ProbSparseSelfAttention follows the Informer paper’s top-query sparsification.
- Decoder self-attention supports causal masks via attn_mask; masks may be 2D or 4D.
- Encoder distillation halves sequence length on intermediate layers via Conv1d+Pool.
"""

import math
import inspect
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# -------------------------
# Utility: causal mask
# -------------------------
def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Creates a lower-triangular mask of shape (seq_len, seq_len),
    where True means 'keep' (visible), and False means 'mask out'.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


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

    def __init__(self, d_model: int, n_heads: int, attention_dropout: float = 0.1, mask_flag: bool = True):
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
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(attention_dropout)

    def forward(self, Q, K, V, attn_mask: torch.Tensor | None = None, causal: bool = False):
        B, L_Q, D = Q.shape
        _, L_K, _ = K.shape
        Q = self.q_proj(Q).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(K).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(V).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, L_Q, D)
        out = self.o_proj(out)
        return self.out_dropout(out)


class ProbSparseSelfAttention(Attention):
    """
    ProbSparse self-attention (Informer).

    Idea: Instead of attending with every query, select the top u queries (u ~ factor*log L)
    most likely to dominate the attention distribution, and compute full attention for them.
    The rest receive a context initialized as the mean of values.

    Masking: Supports optional attn_mask for decoder causal self-attention.
    Accepted mask shapes (boolean True = keep):
    - [L_Q, L_K]
    - [B, H, L_Q, L_K]
    """

    def __init__(self, d_model: int, n_heads: int, attention_dropout: float = 0.1, mask_flag: bool = True, factor: int = 5):
        super().__init__(d_model, n_heads, attention_dropout, mask_flag)
        self.factor = factor

    def _prob_QK(self, Q, K, sample_k: int, n_top: int):
        """
        Core function of ProbSparse attention that identifies the most important
        query positions. Uses a sampling-based approach to approximate query sparsity.

        Shapes
        - Q: [B, H, L_Q, d_k]
        - K: [B, H, L_K, d_k]
        Returns
        - M_top: [B, H, n_top] indices of top query positions per batch/head
        """
        # Extract dimensions
        # B: batch size, H: num heads, L_Q: query length, D: head dimension
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Sample a subset of keys for efficient scoring
        # k: number of keys to sample (min of sample_k or available keys)
        k = min(sample_k, L_K)
        # Randomly select k indices from the key sequence - randperm ensures no duplicates
        perm = torch.randperm(L_K, device=K.device)[:k]
        K_sample = K[:, :, perm, :]

        # Compute attention scores for sampled keys
        # Shape: [B, H, L_Q, k]
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / math.sqrt(D)

        # Compute mean attention score (no gradients needed)
        # This represents the average interaction strength
        with torch.no_grad():
            mean_K = Q_K_sample.mean(dim=-1, keepdim=True)

        # Compute sparsity scores:
        # 1. Use logsumexp for stable probability computation
        # 2. Subtract mean to get relative importance
        M = torch.logsumexp(Q_K_sample, dim=-1) - mean_K.squeeze(-1)

        # Select top-n query positions based on sparsity scores
        # Returns indices of top n_top queries for each batch and head
        M_top = torch.topk(M, n_top, dim=-1)[1]
        return M_top

    def forward(self, Q, K, V, attn_mask: torch.Tensor | None = None):
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

        # ---------------------------------------------------------------------
        # Extract the query vectors corresponding to the top-n most important
        # query positions (as determined by ProbSparse scoring).
        # ---------------------------------------------------------------------
        # Q:       [B, H, L_Q, d_k]   → all query vectors
        # M_top:   [B, H, n_top]      → integer indices of top query positions
        # Goal:    Q_top = [B, H, n_top, d_k] containing only those top queries

        # Step 1: Add a trailing dimension so M_top has the same rank as Q.
        #         Shape: [B, H, n_top, 1]
        M_top_expanded = M_top.unsqueeze(-1)
        # Step 2: Broadcast indices across the feature dimension (d_k)
        #         so we can gather all d_k components for each selected query.
        #         Shape: [B, H, n_top, d_k]
        M_top_expanded = M_top_expanded.expand(-1, -1, -1, self.d_k)
        # Step 3: Gather query vectors at the selected indices along dim=2
        #         (the sequence-length / query-position dimension).
        Q_top = torch.gather(Q, dim=2, index=M_top_expanded)

        # Compute attention scores for the selected queries vs all keys
        attn_scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Handle attention masking for top queries (mask True = keep)
        if attn_mask is not None:
            mask = attn_mask
            if mask.dtype is not torch.bool:
                mask = mask.to(dtype=torch.bool)
            if mask.dim() == 2:
                # [L_Q, L_K] -> [B, H, L_Q, L_K]
                mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, -1, -1)
            # Gather rows for top queries
            mask_top = torch.gather(mask, dim=2, index=M_top.unsqueeze(-1).expand(-1, -1, -1, L_K))
            attn_scores = attn_scores.masked_fill(~mask_top, float('-inf'))
        # Compute attention weights and apply to values
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        # Compute context vectors for top queries
        context_top = torch.matmul(attn, V)

        # ---------------------------------------------------------------------
        # Update the base context tensor with the newly computed attention outputs
        # for the top-n query positions.
        # context:     [B, H, L_Q, d_k]   → initially filled with mean(V)
        # context_top: [B, H, n_top, d_k] → computed attention outputs for top queries
        # ---------------------------------------------------------------------
        context.scatter_(dim=2, index=M_top_expanded, src=context_top)

        # ---------------------------------------------------------------------
        # Reshape the multi-head context back into [B, L_Q, D] for output proj.
        # context: [B, H, L_Q, d_k]
        #   ↓ transpose(1,2): [B, L_Q, H, d_k]
        #   ↓ contiguous() then view: [B, L_Q, D] with D = H*d_k
        # ---------------------------------------------------------------------
        out = context.transpose(1, 2).contiguous().view(B, L_Q, D)
        # Final linear projection (mixes heads)
        return self.out_dropout(self.o_proj(out))


# =====================================================
# Encoder / Decoder Stacks
# =====================================================

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with optional ProbSparse self-attention and distillation.

    - Attention: ProbSparse (default) or full attention via attention_type.
    - FFN: Linear -> GELU -> Linear with residual and LayerNorm (pre/post selectable).
    - Distillation (if enabled): Conv1d+GELU+MaxPool reduces sequence length by ~2.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        distill: bool = True,
        factor: int = 5,
        norm_mode: str = "pre",
        attention_type: str = "prob",
    ):
        super().__init__()
        if attention_type not in ("prob", "full"):
            raise ValueError("attention_type must be 'prob' or 'full'")
        self.attention_type = attention_type
        if attention_type == "prob":
            self.attn = ProbSparseSelfAttention(d_model, n_heads, attention_dropout=dropout, factor=factor)
        else:
            self.attn = Attention(d_model, n_heads, attention_dropout=dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.distill = distill
        self.norm_mode = norm_mode
        if distill:
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
            if self.norm_mode == "post":
                self.norm_after_distill = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.norm_mode == "pre":
            norm_x = self.norm1(x)
            if self.attention_type == "prob":
                x = x + self.dropout(self.attn(norm_x, norm_x, norm_x))
            else:
                x = x + self.dropout(self.attn(norm_x, norm_x, norm_x, causal=False))
            x = x + self.dropout(self.ff(self.norm2(x)))
        else:
            if self.attention_type == "prob":
                sa_out = self.attn(x, x, x)
            else:
                sa_out = self.attn(x, x, x, causal=False)
            x = self.norm1(x + self.dropout(sa_out))
            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout(ff_out))

        if self.distill:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            if self.norm_mode == "post":
                x = self.norm_after_distill(x)
        return x


class DecoderLayer(nn.Module):
    """
    Single layer of the decoder stack with self-attention and cross-attention mechanisms.

    This layer implements a post-norm transformer decoder architecture with both
    masked self-attention and cross-attention to the encoder outputs.

    Architecture:
    ------------
    1. Masked Self-Attention:
       - ProbSparse self-attention with causal masking
       - Pre/Post-normalization for stability
       - Residual connection and dropout

    2. Cross-Attention:
       - Standard attention to encoder outputs
       - Enables information flow from encoder
       - Pre/Post-normalization and residual connection

    3. Feed-Forward:
       - Two linear transformations with GELU
       - Final post-normalization
       - Residual connection and dropout
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, factor: int = 5, norm_mode: str = "pre"):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, attention_dropout=dropout, factor=factor)
        self.cross_attn = Attention(d_model, n_heads, attention_dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_mode = norm_mode

    def forward(self, dec, enc, attn_mask: torch.Tensor | None = None):
        if self.norm_mode == "pre":
            norm_dec_1 = self.norm1(dec)
            dec = dec + self.dropout(self.self_attn(norm_dec_1, norm_dec_1, norm_dec_1, attn_mask=attn_mask))
            dec = dec + self.dropout(self.cross_attn(self.norm2(dec), enc, enc))
            dec = dec + self.dropout(self.ff(self.norm3(dec)))
        else:
            sa_out = self.self_attn(dec, dec, dec, attn_mask=attn_mask)
            dec = self.norm1(dec + self.dropout(sa_out))
            ca_out = self.cross_attn(dec, enc, enc)
            dec = self.norm2(dec + self.dropout(ca_out))
            ff_out = self.ff(dec)
            dec = self.norm3(dec + self.dropout(ff_out))
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
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        distill: bool = True,
        factor: int = 5,
        norm_mode: str = "pre",
        attention_type: str = "prob",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                distill=(distill and i < (n_layers - 1)),
                factor=factor,
                norm_mode=norm_mode,
                attention_type=attention_type,
            )
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model) if norm_mode == "pre" else nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


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
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1, factor: int = 5, norm_mode: str = "pre"):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, factor, norm_mode=norm_mode)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model) if norm_mode == "pre" else nn.Identity()

    def forward(self, dec, enc):
        seq_len = dec.size(1)
        device = dec.device
        attn_mask = generate_causal_mask(seq_len, device)
        for layer in self.layers:
            dec = layer(dec, enc, attn_mask=attn_mask)
        return self.final_norm(dec)


# =====================================================
# Positional & Time Embeddings
# =====================================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    - Precomputes sin/cos table up to max_len and adds to token embeddings.
    - create_with_auto_max_len helper computes a safe max_len from enc+pred lengths.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
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
    def create_with_auto_max_len(cls, d_model: int, enc_len: int, pred_len: int, safety_factor: float = 1.5):
        max_len = int((enc_len + pred_len) * safety_factor)
        return cls(d_model, max_len)


class TimeEmbedding(nn.Module):
    """
    Learnable time embeddings for (hour, weekday, month).
    Each embedding has a small latent dimension, and the combined vector
    is projected to d_model via an MLP projection layer.
    """
    def __init__(self, d_model: int, embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hour_embedding = nn.Embedding(24, embed_dim)
        self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.month_embedding = nn.Embedding(12, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, d_model),
            nn.GELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        if timestamps.size(-1) != 3:
            raise ValueError("timestamps must have shape [B, L, 3] with (hour, weekday, month)")
        hour_emb = self.hour_embedding(timestamps[:, :, 0])
        weekday_emb = self.weekday_embedding(timestamps[:, :, 1])
        month_emb = self.month_embedding(timestamps[:, :, 2])
        time_features = torch.cat([hour_emb, weekday_emb, month_emb], dim=-1)
        time_emb = self.time_mlp(time_features)
        return time_emb


# =====================================================
# Sequence preprocessing helpers
# =====================================================

def compute_log_returns(seq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert level sequences to log returns along the time dimension while preserving shape.
    The first timestep is zero since no previous value exists within the window.
    """
    returns = torch.zeros_like(seq)
    prev = seq[:, :-1, :].clamp_min(eps)
    curr = seq[:, 1:, :].clamp_min(eps)
    returns[:, 1:, :] = torch.log(curr / prev)
    return returns


def build_change_target_from_levels(batch_data: torch.Tensor, asset_index: int, pred_len: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute log returns for the target horizon using level inputs.
    """
    window = batch_data[:, -(pred_len + 1):, asset_index]
    prev = window[:, :-1].clamp_min(eps)
    curr = window[:, 1:].clamp_min(eps)
    return torch.log(curr / prev)


# =====================================================
# Dataset + Data Loading Utilities
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
       - Supports weekday, hour, month embeddings
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

    def __init__(self, data: np.ndarray, timestamps: pd.DatetimeIndex, enc_len: int, pred_len: int):
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

    def _convert_timestamps_to_tensor(self, timestamps: pd.DatetimeIndex) -> torch.Tensor:
        hours = torch.tensor(timestamps.hour.values, dtype=torch.long)
        weekdays = torch.tensor(timestamps.weekday.values, dtype=torch.long)
        months = torch.tensor(timestamps.month.values - 1, dtype=torch.long)
        return torch.stack([hours, weekdays, months], dim=1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with {self.samples} samples")
        seq = self.data[idx: idx + self.enc_len + self.pred_len]
        ts = self.timestamps[idx: idx + self.enc_len + self.pred_len]
        time_tensor = self._convert_timestamps_to_tensor(ts)
        return torch.tensor(seq, dtype=torch.float32), time_tensor


def create_dataloaders(
    df: pd.DataFrame,
    enc_len: int = 96,
    pred_len: int = 1,
    batch_size: int = 32,
    val_batch_size: int = 1,
    val_shuffle: bool = False,
    val_ratio: float = 0.1,
    asset_name: str = "SOL",
):
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
       - StandardScaler normalization (always on)
       - Scaler fit on training data only and applied to validation set
       
    3. Time Features:
       - Extracts timestamps from DataFrame index
       - Preserves temporal information
       - Supports weekday, hour, month embeddings
       
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
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    n = len(df)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:-n_val]
    val_df = df.iloc[-(n_val + enc_len + pred_len):]

    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_scaled = scaler.transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)

    train_ds = TimeSeriesDataset(train_scaled, train_df.index, enc_len, pred_len)
    val_ds = TimeSeriesDataset(val_scaled, val_df.index, enc_len, pred_len)

    def collate_fn(batch):
        # Separate sequences and timestamps
        seqs, times = zip(*batch)
        # Stack them into batches
        seqs = torch.stack(seqs)
        times = torch.stack(times)
        return seqs, times

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=val_shuffle, drop_last=False, collate_fn=collate_fn)
    return train_loader, val_loader, scaler, asset_index


def inverse_transform(y_scaled: np.ndarray, scaler: StandardScaler | None, asset_index: int, n_features: int):
    """Inverse-transform a 1D target back to real scale using a column-aware dummy array."""
    if scaler is None:
        return y_scaled
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, asset_index] = y_scaled
    y_real = scaler.inverse_transform(dummy)[:, asset_index]
    return y_real


def init_weights(module: nn.Module, std: float = 0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        pass

# =====================================================
# Training utilities shared across variants
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
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    min_lr: float = 1e-6
    max_steps: int = 1000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    use_grad_clip: bool = True
    use_amp: bool = True
    use_lr_schedule: bool = True
    patience: int = 5
    min_delta: float = 0.0001
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
    """
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer


def get_lr(step: int, cfg: TrainConfig):
    """Warmup followed by cosine decay down to min_lr (if schedule enabled)."""
    if not cfg.use_lr_schedule:
        return cfg.learning_rate
    if step < cfg.warmup_steps:
        return cfg.learning_rate * float(step) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * cosine


# =====================================================
# Generic training loop
# =====================================================

def train_model_generic(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    target_fn,
    *,
    target_kwargs: dict | None = None,
    logger=None,
    log_interval: int = 10,
    val_interval: int = 100,
):
    """
    Generic training loop shared by both model variants.

    Parameters
    - model: nn.Module that maps (batch_data, timestamps) -> y_pred
    - train_loader/val_loader: yield (data, timestamps)
    - cfg: TrainConfig controlling optimization, AMP, schedule, early stopping
    - target_fn: callable(batch_data, model, **target_kwargs) -> y_true tensor
      Enc-dec variant: y_true = batch[:, -pred_len:, asset_index]
      Enc-only variant: y_true depends on target_type (price|change)
    - logger: optional Python logger
    - log_interval/val_interval: step intervals for progress and validation
    Returns a dict with training/validation curves and best checkpoint info.
    """
    if target_kwargs is None:
        target_kwargs = {}

    if logger is None:
        import logging as _logging
        logger = _logging.getLogger("train")

    device = torch.device(cfg.device)
    model = model.to(device)
    opt = configure_optimizers(model, weight_decay=cfg.weight_decay, learning_rate=cfg.learning_rate, device_type=cfg.device)
    scaler = torch.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    loss_fn = nn.MSELoss()

    step = 0
    best_val_loss = float("inf")
    best_val_step = 0
    patience_counter = 0
    best_model_state = None
    early_stop = False

    train_losses, val_losses, steps = [], [], []
    start_time = time.time()

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for data, ts in loader:
                data, ts = data.to(device), ts.to(device)
                y_true = target_fn(data, model, **target_kwargs)
                y_pred = model(data, ts)
                losses.append(loss_fn(y_pred, y_true).item())
        return float(np.mean(losses)) if losses else float("nan")

    model.train()
    while step < cfg.max_steps and not early_stop:
        for batch_data, batch_timestamps in train_loader:
            batch_data, batch_timestamps = batch_data.to(device), batch_timestamps.to(device)
            y_true = target_fn(batch_data, model, **target_kwargs)

            lr = get_lr(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                y_pred = model(batch_data, batch_timestamps)
                loss = loss_fn(y_pred, y_true)

            opt.zero_grad()
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

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                samples_per_s = batch_data.size(0) / max(elapsed, 1e-9)
                logger.info(f"[Step {step:5d}] train_loss={loss.item():.6f} | lr={lr:.3e} | samples/s={samples_per_s:.1f}")
                train_losses.append(loss.item())
                steps.append(step)
                start_time = time.time()

            if step % val_interval == 0 and step > 0:
                val_loss = evaluate(val_loader)
                logger.info(f"[Step {step:5d}] val_loss={val_loss:.6f}")
                val_losses.append(val_loss)
                if val_loss + cfg.min_delta < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = step
                    patience_counter = 0
                    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.patience:
                        early_stop = True
                        logger.info("Early stopping triggered.")
                        break

            step += 1
            if step >= cfg.max_steps:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from step {best_val_step} with val_loss={best_val_loss:.6f}")

    return  model, steps,train_losses,val_losses,best_val_loss, best_val_step
