
"""
Encoder-Only Informer for 1-Step-Ahead Forecasting
===================================================
- Keeps your original utilities (logging, dataset, optimizer, LR schedule, plotting).
- Removes decoder/guiding window and uses an encoder-only Informer.
- Adds config toggles:
    - attention_type: "prob" (ProbSparse) or "full" (standard MHA)
    - target_type:    "change" (log-percent change) or "price" (level)
- Predicts 1-step ahead only (pred_len must be 1).
- Optional TimeEmbedding retained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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
log = logging.getLogger("informer_encoder_only")

# =====================================================
# Attention Mechanisms
# =====================================================

class Attention(nn.Module):
    """
    Standard Multi-Head Attention (full attention).
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
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(attention_dropout)

    def forward(self, Q, K, V, attn_mask=None, causal=False):
        B, L_Q, D = Q.shape
        _, L_K, _ = K.shape
        Q = self.q_proj(Q).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(K).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(V).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dropout_p, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, L_Q, D)
        out = self.o_proj(out)
        return self.out_dropout(out)

class ProbSparseSelfAttention(Attention):
    """
    ProbSparse Self-Attention (approximate attention over dominant queries).
    """
    def __init__(self, d_model, n_heads, attention_dropout=0.1, mask_flag=True, factor=5):
        super().__init__(d_model, n_heads, attention_dropout, mask_flag)
        self.factor = factor

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape
        k = min(sample_k, L_K)
        perm = torch.randperm(L_K, device=K.device)[:k]
        K_sample = K[:, :, perm, :]
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / math.sqrt(D)
        with torch.no_grad():
            mean_K = Q_K_sample.mean(dim=-1, keepdim=True)
        M = torch.logsumexp(Q_K_sample, dim=-1) - mean_K.squeeze(-1)
        M_top = torch.topk(M, n_top, dim=-1)[1]
        return M_top

    def forward(self, Q, K, V):
        B, L_Q, D = Q.shape
        _, L_K, _ = K.shape
        Q = self.q_proj(Q).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(K).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(V).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        factor = self.factor
        sample_k = max(1, int(factor * math.log(max(L_K, 2))))
        n_top = min(L_Q, max(1, int(factor * math.log(max(L_Q, 2)))))

        M_top = self._prob_QK(Q, K, sample_k, n_top)

        with torch.no_grad():
            context = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1).clone()

        M_top_expanded = M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        Q_top = torch.gather(Q, dim=2, index=M_top_expanded)

        attn_scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        context_top = torch.matmul(attn, V)

        context.scatter_(dim=2, index=M_top_expanded, src=context_top)

        out = context.transpose(1, 2).contiguous().view(B, L_Q, D)
        return self.o_proj(out)

# =====================================================
# Encoder
# =====================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, distill=True, factor=5,
                 norm_mode="pre", attention_type="prob"):
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
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
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

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1,
                 distill=True, factor=5, norm_mode="pre", attention_type="prob"):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, distill=(distill and i < (n_layers - 1)),
                         factor=factor, norm_mode=norm_mode, attention_type=attention_type)
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model) if norm_mode == "pre" else nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

# Positional & Time Embeddings
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
        max_len = int((enc_len + pred_len) * safety_factor)
        return cls(d_model, max_len)

class TimeEmbedding(nn.Module):
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
# Encoder-Only Forecasting Model
# =====================================================

class StockInformerEncoderOnly(nn.Module):
    """
    Encoder-only Informer that predicts 1-step ahead.
    - attention_type: "prob" or "full"
    - target_type: "change" (log-return) or "price"
    """
    def __init__(self, config, asset_index=0):
        super().__init__()
        # Required config
        self.asset_index = asset_index
        self.d_input = config["d_input"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_ff = config["d_ff"]
        self.enc_layers = config["enc_layers"]
        self.dropout = config["dropout"]
        self.distill = config["distill"]
        self.factor = config["factor"]
        self.use_time_embedding = config.get("use_time_embedding", False)
        self.norm_mode = config.get("norm_mode", "pre")
        self.attention_type = config.get("attention_type", "prob")  # "prob" | "full"
        self.target_type = config.get("target_type", "change")      # "change" | "price"

        # Sequence structure
        self.enc_len = config["enc_len"]
        self.pred_len = config.get("pred_len", 1)
        if self.pred_len != 1:
            raise ValueError("Encoder-only implementation supports pred_len=1 only.")

        # Embeddings
        self.enc_embedding = nn.Linear(self.d_input, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model)
        if self.use_time_embedding:
            self.time_embedding = TimeEmbedding(self.d_model, embed_dim=8, dropout=self.dropout)
            self.time_add_norm = nn.LayerNorm(self.d_model)

        # Encoder backbone
        self.encoder = Encoder(
            self.d_model, self.n_heads, self.d_ff, self.enc_layers,
            dropout=self.dropout, distill=self.distill,
            factor=self.factor, norm_mode=self.norm_mode,
            attention_type=self.attention_type
        )

        # Output projection
        self.proj = nn.Linear(self.d_model, 1)

    def forward(self, seq, timestamps=None):
        """
        seq: [B, L_total, d_input] with L_total = enc_len + pred_len
        timestamps: [B, L_total, 3] (hour, weekday, month) if use_time_embedding=True
        """
        L_total = seq.size(1)
        if L_total < self.enc_len + self.pred_len:
            raise ValueError(f"Input too short ({L_total}); need >= {self.enc_len + self.pred_len}")

        # Encoder input
        enc_x = seq[:, :self.enc_len, :]  # [B, enc_len, d_input]
        enc_h = self.pos_enc(self.enc_embedding(enc_x))

        if self.use_time_embedding and timestamps is not None:
            enc_ts = timestamps[:, :self.enc_len]
            enc_h = self.time_add_norm(enc_h + self.time_embedding(enc_ts))

        enc_out = self.encoder(enc_h)  # [B, L', d_model] (L' may be reduced by distill)

        # Use the last timestep representation
        last_h = enc_out[:, -1, :]  # [B, d_model]
        out = self.proj(last_h)     # [B, 1]

        # Return shape [B, pred_len] to be compatible with your training loop
        return out.squeeze(-1).unsqueeze(-1)

# =====================================================
# Dataset and Data Loading Utilities
# =====================================================

class TimeSeriesDataset(Dataset):
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
        hours = torch.tensor(timestamps.hour.values, dtype=torch.long)
        weekdays = torch.tensor(timestamps.weekday.values, dtype=torch.long)
        months = torch.tensor(timestamps.month.values - 1, dtype=torch.long)
        time_tensor = torch.stack([hours, weekdays, months], dim=1)
        return time_tensor

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with {self.samples} samples")
        seq = self.data[idx: idx + self.enc_len + self.pred_len]
        if len(seq) != self.enc_len + self.pred_len:
            raise RuntimeError(f"Unexpected sequence length {len(seq)}, expected {self.enc_len + self.pred_len}")
        ts = self.timestamps[idx: idx + self.enc_len + self.pred_len]
        time_tensor = self._convert_timestamps_to_tensor(ts)
        return torch.tensor(seq, dtype=torch.float32), time_tensor

def create_dataloaders(df, enc_len=96, pred_len=1, batch_size=32, val_batch_size=1, val_shuffle=False, val_ratio=0.1, asset_name="SOL"):
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
        seqs, times = zip(*batch)
        seqs = torch.stack(seqs)
        times = torch.stack(times)
        return seqs, times

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=val_shuffle, drop_last=False, collate_fn=collate_fn)
    return train_loader, val_loader, scaler, asset_index

def inverse_transform(y_scaled, scaler, asset_index, n_features):
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, asset_index] = y_scaled
    y_real = scaler.inverse_transform(dummy)[:, asset_index]
    return y_real

# =====================================================
# Optimized Training Utilities (kept; minor tweaks to handle target_type)
# =====================================================

@dataclass
class TrainConfig:
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
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    log.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    log.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
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
    if not cfg.use_lr_schedule:
        return cfg.learning_rate
    if step < cfg.warmup_steps:
        return cfg.learning_rate * float(step) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * cosine

# Helper: build y_true for both target types
def build_target(batch_data: torch.Tensor, asset_index: int, target_type: str) -> torch.Tensor:
    """
    batch_data: [B, enc_len + 1, d_input], pred_len must be 1
    Returns: [B, 1]
    """
    if batch_data.size(1) < 2:
        raise ValueError("Need at least 2 time steps to compute change target.")
    if target_type == "price":
        y = batch_data[:, -1, asset_index]  # scaled price target
        return y.unsqueeze(-1)
    elif target_type == "change":
        # log-percent change between last encoder step and target step
        prev = batch_data[:, -2, asset_index].clamp_min(1e-8)
        curr = batch_data[:, -1, asset_index].clamp_min(1e-8)
        y = torch.log(curr / prev)
        return y.unsqueeze(-1)
    else:
        raise ValueError("target_type must be 'price' or 'change'")

# === Training Loop ===
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    asset_index: int = 0,
    log_interval: int = 10,
    val_interval: int = 100,
):
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

    def evaluate(loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for data, ts in loader:
                data, ts = data.to(device), ts.to(device)
                y_true = build_target(data, asset_index, getattr(model, "target_type", "change"))
                y_pred = model(data, ts)  # [B, 1]
                losses.append(loss_fn(y_pred, y_true).item())
        return float(np.mean(losses)) if losses else float("nan")

    model.train()
    while step < cfg.max_steps and not early_stop:
        for batch_data, batch_timestamps in train_loader:
            batch_data, batch_timestamps = batch_data.to(device), batch_timestamps.to(device)
            y_true = build_target(batch_data, asset_index, getattr(model, "target_type", "change"))  # [B, 1]

            lr = get_lr(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                y_pred = model(batch_data, batch_timestamps)  # [B, 1]
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
                log.info(f"[Step {step:5d}] train_loss={loss.item():.6f} | lr={lr:.3e} | samples/s={samples_per_s:.1f}")
                train_losses.append(loss.item())
                steps.append(step)
                start_time = time.time()

            if val_loader and step % val_interval == 0:
                val_loss = evaluate(val_loader)
                val_losses.append((step, val_loss))
                improved = val_loss < (best_val_loss - cfg.min_delta)
                if improved:
                    best_val_loss = val_loss
                    best_val_step = step
                    patience_counter = 0
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    log.info(f"[Val {step:5d}] loss={val_loss:.6f} | best={best_val_loss:.6f} | patience={patience_counter}/{cfg.patience}")
                    if patience_counter >= cfg.patience:
                        log.info(f"Early stopping triggered at step {step}.")
                        early_stop = True
                        break
                model.train()

            step += 1
            if step >= cfg.max_steps or early_stop:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, steps, best_val_loss, best_val_step

def init_weights(module: nn.Module, std: float = 0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        pass

# =====================================================
# Example configuration/test run when run as script
# =====================================================
if __name__ == "__main__":
    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    # Change this path to your CSV
    csv_path = r"D:\Quan\Quants\Neural Network\financial_attention\1h_data_20220101_20250601.csv"
    closes = pd.read_csv(csv_path, index_col=0, parse_dates=True)[['SOL', 'ETH', 'BTC','ADA','XRP','LTC','TRX','LINK','DOT','DOGE']]

    # -----------------------------
    # Config
    # -----------------------------
    config = {
        "d_input": len(closes.columns),
        "d_model": 16,
        "n_heads": 4,
        "d_ff": 32,
        "enc_layers": 3,
        "dropout": 0.0,
        "distill": True,
        "enc_len": 96,
        "pred_len": 1,          # MUST be 1 in this implementation
        "factor": 5,
        "use_time_embedding": True,
        "norm_mode": "pre",
        # New toggles:
        "attention_type": "prob",   # "prob" or "full"
        "target_type": "change",    # "change" or "price"
    }

    train_loader, val_loader, scaler, asset_idx = create_dataloaders(
        closes, enc_len=config["enc_len"], pred_len=config["pred_len"],
        batch_size=32, val_ratio=0.1, asset_name="SOL"
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = StockInformerEncoderOnly(config, asset_index=asset_idx)
    model.apply(lambda m: init_weights(m))

    # -----------------------------
    # Training config
    # -----------------------------
    tcfg = TrainConfig(learning_rate=1e-4, weight_decay=0.01, max_steps=200, warmup_steps=10, use_amp=False, device="cpu")

    model, train_hist, val_hist, steps_hist, best_val_loss, best_val_step = train_model(
        model, train_loader, val_loader, tcfg, asset_index=asset_idx
    )

    total_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_learnable_params:,}")

    # -----------------------------
    # Validation predictions & plotting
    # -----------------------------
    device = torch.device(tcfg.device)
    model = model.to(device)
    preds = []
    targets = []
    with torch.no_grad():
        for val_seqs, val_times in val_loader:
            val_seqs = val_seqs.to(device)
            val_times = val_times.to(device)
            y_pred = model(val_seqs, val_times)  # [B, 1]
            y_true = build_target(val_seqs, asset_idx, model.target_type)  # [B, 1]
            preds.append(y_pred.cpu().numpy())
            targets.append(y_true.cpu().numpy())
    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()

    if config["target_type"] == "price":
        preds_plot = inverse_transform(preds, scaler, asset_idx, config["d_input"])
        targets_plot = inverse_transform(targets, scaler, asset_idx, config["d_input"])
        y_label = "Price"
        title = "Predicted vs Actual Price (Validation Set)"
    else:
        preds_plot = preds
        targets_plot = targets
        y_label = "Log Return (1h)"
        title = "Predicted vs Actual Log-Return (Validation Set)"

    df_plot = pd.DataFrame({
        "Time Step": np.arange(len(targets_plot)),
        "Actual": targets_plot,
        "Predicted": preds_plot
    })
    df_plot = df_plot.melt(id_vars="Time Step", value_vars=["Actual", "Predicted"],
                           var_name="Type", value_name=y_label)

    fig = px.line(df_plot, x="Time Step", y=y_label, color="Type", title=title)
    fig.show()
