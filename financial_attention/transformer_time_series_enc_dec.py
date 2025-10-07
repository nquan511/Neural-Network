# informer_forecaster.py
# =====================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =====================================================
# ProbSparse Attention
# =====================================================
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_dropout=0.1, mask_flag=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.mask_flag = mask_flag

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # importance estimation
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K[:, :, index_sample, :]
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / math.sqrt(D)
        M = torch.logsumexp(Q_K_sample, dim=-1) - torch.mean(Q_K_sample, dim=-1)
        M_top = torch.topk(M, n_top, dim=-1)[1]
        return M_top

    def forward(self, Q, K, V, attn_mask=None):
        B, L_Q, D = Q.shape
        _, L_K, _ = K.shape

        Q = self.q_proj(Q).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(K).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(V).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        u = max(1, int(0.5 * math.log(L_K)))  # hyperparameter
        n_top = max(1, int(L_Q * math.log(L_K)))
        M_top = self._prob_QK(Q, K, u, n_top)

        context = torch.mean(V, dim=-2, keepdim=True).expand(-1, -1, L_Q, -1).clone()
        Q_reduce = Q.gather(2, M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k))
        attn = torch.matmul(Q_reduce, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        context_top = torch.matmul(attn, V)
        context.scatter_(2, M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k), context_top)

        out = context.transpose(1, 2).contiguous().view(B, L_Q, D)
        return self.o_proj(out)


# =====================================================
# Encoder and Decoder Layers
# =====================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, distill=True):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads)
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
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads)
        self.cross_attn = ProbSparseSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec, enc):
        dec = dec + self.dropout(self.self_attn(dec, dec, dec))
        dec = self.norm1(dec)
        dec = dec + self.dropout(self.cross_attn(dec, enc, enc))
        dec = self.norm2(dec)
        dec = dec + self.dropout(self.ff(dec))
        dec = self.norm3(dec)
        return dec


# =====================================================
# Encoder / Decoder Stacks
# =====================================================
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1, distill=True):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, distill)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, dec, enc):
        for layer in self.layers:
            dec = layer(dec, enc)
        return dec


# =====================================================
# Positional Encoding
# =====================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =====================================================
# Full Informer Forecast Model
# =====================================================
class InformerForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_input = config["d_input"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_ff = config["d_ff"]
        self.enc_layers = config["enc_layers"]
        self.dec_layers = config["dec_layers"]
        self.dropout = config["dropout"]
        self.distill = config["distill"]

        self.enc_embedding = nn.Linear(self.d_input, self.d_model)
        self.dec_embedding = nn.Linear(self.d_input, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model)

        self.encoder = Encoder(self.d_model, self.n_heads, self.d_ff, self.enc_layers,
                               dropout=self.dropout, distill=self.distill)
        self.decoder = Decoder(self.d_model, self.n_heads, self.d_ff, self.dec_layers,
                               dropout=self.dropout)
        self.proj = nn.Linear(self.d_model, 1)

        self.enc_len = config["enc_len"]
        self.guiding_len = config["guiding_len"]
        self.pred_len = config["pred_len"]

    def forward(self, seq):
        """
        seq: [B, L_total, d_input]
        Must have L_total >= enc_len + pred_len
        """
        B = seq.size(0)

        enc_x = seq[:, :self.enc_len, :]
        enc_h = self.pos_enc(self.enc_embedding(enc_x))
        enc_out = self.encoder(enc_h)

        # Decoder input = guiding context + zeros
        dec_context = seq[:, self.enc_len - self.guiding_len : self.enc_len, :]
        dec_placeholder = torch.zeros(B, self.pred_len, self.d_input, device=seq.device)
        dec_input = torch.cat([dec_context, dec_placeholder], dim=1)
        dec_h = self.pos_enc(self.dec_embedding(dec_input))
        dec_out = self.decoder(dec_h, enc_out)

        # Only predict on placeholder region
        out = self.proj(dec_out[:, -self.pred_len:, :])
        return out.squeeze(-1)


# =====================================================
# Example configuration
# =====================================================
if __name__ == "__main__":
    config = {
        "d_input": 4,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 256,
        "enc_layers": 3,
        "dec_layers": 2,
        "dropout": 0.1,
        "distill": True,
        "enc_len": 96,
        "guiding_len": 48,
        "pred_len": 24,
    }

    model = InformerForecaster(config)
    x = torch.randn(8, 120, 4)
    y_hat = model(x)
    print("Output:", y_hat.shape)  # [B, pred_len]
