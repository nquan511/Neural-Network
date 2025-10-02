"""
gpt_timeseries.py
==================

Core design decisions:
- Input at time t:  x_t = [ z_{t-1}, ETH_t, BTC_t ]  (continuous floats)
- Target at time t: y_t = z_t (SOL return)
- Input projection: a linear layer projects (d_input=3) -> n_embd
- Positional embeddings preserved (nn.Embedding)
- Multi-head causal attention uses F.scaled_dot_product_attention(is_causal=True)
- Output head: linear projection n_embd -> 1
- Loss: MSE (regression)
- Optimizer: AdamW with decoupled weight decay grouping
- DataLoader: robust sequence sampler that produces (B, T, d_input) and (B, T) targets

Optimization tricks (configurable):
- weight_decay (decoupled AdamW)
- cosine LR schedule with warmup
- gradient clipping
- AMP mixed precision
All OFF by default for clean baselines.

"""

from dataclasses import dataclass
import pandas as pd
import math
import time
import typing as tp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import logging
import plotly.express as px

# ---------------------------------------------------------
# Logging configuration (production-friendly)
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("gpt_timeseries")


# ---------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------
@dataclass
class GPTTSConfig:
    # model architecture
    block_size: int = 1024        # max context length (T)
    n_layer: int = 12             # number of transformer blocks
    n_head: int = 12              # number of attention heads
    n_embd: int = 768             # embedding dimension (model width)
    d_input: int = 3              # number of continuous features per time step (z_{t-1}, ETH, BTC)
    d_ff: int = 4 * 768           # feedforward inner dimension
    dropout: float = 0.0          # dropout in attention (kept default 0 for stability)
    # training
    weight_decay: float = 0.1
    learning_rate: float = 3e-4
    grad_clip: float = 1.0

    # optimization tricks (default OFF for ablation experiments)
    use_weight_decay: bool = True
    use_lr_schedule: bool = False
    use_grad_clip: bool = True
    use_amp: bool = True

    # numeric
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_conv_qk: bool = False
    conv_kernel_size: int = 3


# ---------------------------------------------------------
# Helper: weight initialization
# ---------------------------------------------------------
def _init_weights(module: nn.Module, std: float = 0.02, n_layer: int = 12):
    """
    Weight initialization similar to miniGPT/GPT2-style.
    Linear weights normal(mean=0, std) and embeddings likewise.
    For certain final projections inside residual stream we scale init down.
    """
    if isinstance(module, nn.Linear):
        local_std = std
        # detect special scaling attribute (some linear layers might set this)
        if hasattr(module, "NANOGPT_SCALE_INIT") and getattr(module, "NANOGPT_SCALE_INIT"):
            local_std = std * (2 * n_layer) ** -0.5
        nn.init.normal_(module.weight, mean=0.0, std=local_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        # default PyTorch LayerNorm is already fine, leave as-is
        pass


# ---------------------------------------------------------
# Causal Self-Attention with flash attention
# ---------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention using PyTorch's efficient
    scaled_dot_product_attention primitive (supports is_causal=True).
    This is equivalent to the standard QKV / softmax attention but
    uses optimized kernels where available.
    """

    def __init__(self, config: GPTTSConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_conv_qk = getattr(config, "use_conv_qk", False)
        self.conv_kernel_size = getattr(config, "conv_kernel_size", 3)

        if self.use_conv_qk:
            # Causal conv for Q and K
            self.q_conv = nn.Conv1d(
                config.n_embd, config.n_embd,
                kernel_size=self.conv_kernel_size,
                padding=0, bias=False
            )
            self.k_conv = nn.Conv1d(
                config.n_embd, config.n_embd,
                kernel_size=self.conv_kernel_size,
                padding=0, bias=False
            )
            # V is still linear
            self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        else:
            # Standard fused QKV projection
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True

        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_embd)
        Returns:
            y: (B, T, n_embd)
        """
        B, T, C = x.size()
        if self.use_conv_qk:
            # x: (B, T, C) -> (B, C, T) for conv1d
            x_t = x.transpose(1, 2)
            # Causal padding: pad left only
            pad = self.conv_kernel_size - 1
            x_pad = F.pad(x_t, (pad, 0))
            q = self.q_conv(x_pad).transpose(1, 2)  # (B, T, C)
            k = self.k_conv(x_pad).transpose(1, 2)  # (B, T, C)
            v = self.v_proj(x)                      # (B, T, C)
        else:
            qkv = self.c_attn(x)  # (B, T, 3*C)
            q, k, v = qkv.split(self.n_embd, dim=2)  # (B, T, C) each

        # reshape into heads: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


# ---------------------------------------------------------
# MLP / Feed-forward
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: GPTTSConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: GPTTSConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm residual block
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------
# GPTTimeSeries model
# ---------------------------------------------------------
class GPTTimeSeries(nn.Module):
    """
    GPT-style Transformer adapted for continuous time-series regression.

    Interface differences from text GPT:
      - Input: float tensor (B, T, d_input)
      - Input projection: Linear(d_input -> n_embd)
      - Positional embeddings: preserved (nn.Embedding)
      - Output head: Linear(n_embd -> 1) (predict scalar per time-step)
      - Loss: MSE (computed externally or by model.forward(targets=...))

    Usage:
      model = GPTTimeSeries(cfg)
      logits, loss = model(x, targets)   # x: (B,T,d_input), targets: (B,T)
    """

    def __init__(self, config: GPTTSConfig):
        super().__init__()
        self.config = config

        # continuous input projection
        self.input_proj = nn.Linear(config.d_input, config.n_embd)

        # positional embeddings (learnable) same as GPT
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)
        # transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # final regression head (predict 1 scalar per time step)
        self.lm_head = nn.Linear(config.n_embd, 1)

        # initialize weights
        self.apply(lambda module: _init_weights(module, std=1/math.sqrt(config.n_embd), n_layer=config.n_layer))

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, d_input) float input features
            targets: optional (B, T) float targets (z_t) to compute MSE loss

        Returns:
            logits: (B, T, 1) predicted next-step values
            loss: scalar MSE loss if targets provided
        """
        B, T, d_in = x.size()
        assert d_in == self.config.d_input, f"Expected d_input={self.config.d_input}, got {d_in}"
        assert T <= self.config.block_size, f"Sequence length T={T} exceeds block_size={self.config.block_size}"

        # input projection + positional embedding
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        h = self.input_proj(x) + pos_emb.unsqueeze(0)  # broadcast across batch

        h = self.drop(h)

        for block in self.h:
            h = block(h)

        h = self.ln_f(h)

        logits = self.lm_head(h)  # (B, T, 1)

        loss = None
        if targets is not None:
            # targets shape: (B, T) -> match logits
            loss = F.mse_loss(logits.squeeze(-1), targets)

        return logits, loss

    # training helper: create optimizer with decay/no-decay groups (same logic as original)
    def configure_optimizers(self, weight_decay: float, learning_rate: float, device_type: str = "cpu"):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
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


# ---------------------------------------------------------
# Dataset and collate helper
# ---------------------------------------------------------
class TimeSeriesDataset:
    """
    Lightweight dataset that creates autoregressive training examples.

    Input format expected:
      data: np.ndarray or torch.Tensor of shape (N, C) where columns = [SOL, ETH, BTC,...]

    For sequence modeling we generate sequences of length T:
      For t in [0 .. T-1] of a sequence starting at index s:
        input[t]  = [ z_{s+t-1}, ETH_{s+t}, BTC_{s+t} ]    <-- note z_{s-1} for first input must exist
        target[t] = z_{s+t}

    To keep things simple, we only sample starting indices where the lag exists (i.e. start >= 1).
    """

    def __init__(
        self,
        data: tp.Union[np.ndarray, torch.Tensor],
        block_size: int,
        split: str = 'train',
        train_frac: float = 0.9,
        target_col: int = 0,
        feature_cols: tp.Optional[tp.List[int]] = None,
    ):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        assert data.ndim == 2 and data.shape[1] >= 1, "data must be shape (N, C), C>=1"
        self.raw = data
        assert split in {'train', 'val'}
        n = int(train_frac * len(self.raw))
        if split == 'train':
            self.data = self.raw[:n]
        else:
            self.data = self.raw[n:]
        self.block_size = block_size
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols is not None else list(range(data.shape[1]))

        # For lag, we require s >= 1
        self.valid_starts = list(range(1, max(1, len(self.data) - block_size + 1)))
        if not self.valid_starts:
            raise ValueError("Not enough data to form one training sequence. Decrease block_size or provide more data.")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single sequence (x_seq, y_seq)
        where:
          x_seq shape: (T, d_input)
          y_seq shape: (T,)
        """
        s = self.valid_starts[idx]
        T = self.block_size
        seq = self.data[s - 1: s + T]  # (T+1, C)
        # Lagged target
        z_lag = seq[:-1, self.target_col:self.target_col+1]  # (T, 1)
        # Features at time t
        covars = seq[1:, self.feature_cols]  # (T, d_covars)
        x = torch.cat([z_lag, covars], dim=-1) if self.target_col not in self.feature_cols else covars
        y = seq[1:, self.target_col]  # (T,)
        return x, y

    @staticmethod
    def collate_fn(batch: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a list of (x_seq, y_seq) into batched tensors (B, T, d_input) and (B, T)
        """
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.stack([b[1] for b in batch], dim=0)
        return xs, ys

# ---------------------------------------------------------
# Training loop with toggles
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, cfg, max_steps=1000, warmup_steps=100):
    device = torch.device(cfg.device)
    model.to(device)
    opt = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, device_type=cfg.device)

    scaler = torch.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    step, start = 0, time.time()
    best_val_loss = float('inf')
    best_model_state = None

    train_loss_history = []
    val_loss_history = []
    steps_history = []

    while step < max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            lr = cfg.learning_rate  # or use get_lr(step) if using LR schedule
            for g in opt.param_groups: g['lr'] = lr

            opt.zero_grad()
            ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.use_amp)
            with ctx:
                _, loss = model(x, y)

            if cfg.use_amp:
                scaler.scale(loss).backward()
                if cfg.use_grad_clip: scaler.unscale_(opt)
            else:
                loss.backward()

            if cfg.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if cfg.use_amp:
                scaler.step(opt); scaler.update()
            else:
                opt.step()

            if step % 10 == 0:
                tok_s = (x.shape[0] * x.shape[1]) / max(1e-9, time.time() - start)
                log.info(f"step {step} | loss {loss.item():.6f} | lr {lr:.3e} | tok/s {tok_s:.1f}")
                train_loss_history.append(loss.item())
                steps_history.append(step)

            # --- Validation loss and checkpointing ---
            if step % 100 == 0 and val_loader is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        _, vloss = model(vx, vy)
                        val_losses.append(vloss.item())
                mean_val_loss = np.mean(val_losses)
                log.info(f"step {step} | VALIDATION loss {mean_val_loss:.6f}")
                val_loss_history.append((step, mean_val_loss))
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    log.info(f"New best model at step {step} (val loss {best_val_loss:.6f})")
                model.train()

            step += 1
            if step >= max_steps: break

    if best_model_state is not None:
        torch.save(best_model_state, "gpt_ts_best.pth")
        log.info("Best model saved to gpt_ts_best.pth")
    return model, train_loss_history, val_loss_history, steps_history


# ---------------------------------------------------------
# Convenience loader builder for your CSV -> returns -> DataLoader
# ---------------------------------------------------------
def build_loaders_from_returns(
    returns_array: np.ndarray,
    block_size: int,
    batch_size: int,
    train_frac: float = 0.9,
    num_workers: int = 0,
    target_col: int = 0,
    feature_cols: tp.Optional[tp.List[int]] = None,
):
    train_ds = TimeSeriesDataset(
        returns_array, block_size=block_size, split='train', train_frac=train_frac,
        target_col=target_col, feature_cols=feature_cols
    )
    val_ds = TimeSeriesDataset(
        returns_array, block_size=block_size, split='val', train_frac=train_frac,
        target_col=target_col, feature_cols=feature_cols
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                               collate_fn=TimeSeriesDataset.collate_fn, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                             collate_fn=TimeSeriesDataset.collate_fn, num_workers=num_workers)
    return train_loader, val_loader


# ---------------------------------------------------------
# Example script usage (if run as main)
# ---------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    csv_path = r"D:\Quan\Quants\Neural Network\financial_attention\1h_data_20220101_20250601.csv"
    log.info("Loading CSV...")
    closes = pd.read_csv(csv_path, index_col=0, parse_dates=True)[['SOL', 'ETH', 'BTC']]
    rets = np.log(closes).diff().dropna().values.astype(np.float32)  # shape (N, 3)

    # -----------------------------
    # Config + model
    # -----------------------------
    target_col = 2  # BTC
    feature_cols = [0, 1, 2]  # SOL, ETH, BTC

    cfg = GPTTSConfig(
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        d_input=len(feature_cols) + 1,  # +1 if you include lagged target
        dropout=0.0,
        learning_rate=3e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = GPTTimeSeries(cfg)

    # -----------------------------
    # Build loaders
    # -----------------------------
    B = 8
    T = cfg.block_size
    train_loader, val_loader = build_loaders_from_returns(
        rets, block_size=cfg.block_size, batch_size=B,
        target_col=target_col, feature_cols=feature_cols
    )

    # -----------------------------
    # Train
    # -----------------------------
    model = train_model(model, train_loader, val_loader, cfg, max_steps=100, warmup_steps=0)
    # -----------------------------
    # Plot predicted vs actual returns on validation set (Plotly)
    # -----------------------------
    model.eval()
    device = torch.device(cfg.device)
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            out, _ = model(x)
            preds.append(out.squeeze(-1).cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()

    df_plot = pd.DataFrame({
        "Time Step": np.arange(len(targets)),
        "Actual Returns": targets,
        "Predicted Returns": preds
    })
    df_plot = df_plot.melt(id_vars="Time Step", value_vars=["Actual Returns", "Predicted Returns"],
                           var_name="Type", value_name="Return")

    fig = px.line(df_plot, x="Time Step", y="Return", color="Type",
                  title="Predicted vs Actual Returns (Validation Set)")
    fig.show()
    
    # model can be saved with torch.save(model.state_dict(), 'gpt_ts.pth')
    log.info("Done")

