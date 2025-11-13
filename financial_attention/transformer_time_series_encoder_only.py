
"""
Encoder-Only Informer for 1-Step-Ahead Forecasting (refactored)
================================================================
- Imports shared attention/encoder/embeddings/dataloaders/training utils
- Keeps only the model wrapper, build_target, and training loop specifics
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from components import (
    Encoder,
    PositionalEncoding,
    TimeEmbedding,
    TrainConfig,
    train_model_generic,
    build_change_target_from_levels,
)


logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger("informer_encoder_only")


class StockInformerEncoderOnly(nn.Module):
    """
    Encoder-only Informer that predicts 1-step ahead.
    - attention_type: "prob" or "full"
    - target_type: "change" (log-return) or "price"
    """

    def __init__(self, config, asset_index: int = 0):
        super().__init__()
        self.asset_index = asset_index
        self.d_input = config["d_input"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_ff = config["d_ff"]
        self.enc_layers = config["enc_layers"]
        self.dropout = config["dropout"]
        self.distill = config["distill"]
        self.factor = config.get("factor", 5)
        self.use_time_embedding = config.get("use_time_embedding", False)
        self.norm_mode = config.get("norm_mode", "pre")
        self.attention_type = config.get("attention_type", "prob")
        self.target_type = config.get("target_type", "change")

        self.enc_len = config["enc_len"]
        self.pred_len = config.get("pred_len", 1)
        if self.pred_len != 1:
            raise ValueError("Encoder-only implementation supports pred_len=1 only.")

        self.enc_embedding = nn.Linear(self.d_input, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model)
        if self.use_time_embedding:
            self.time_embedding = TimeEmbedding(self.d_model, embed_dim=8, dropout=self.dropout)
            self.time_add_norm = nn.LayerNorm(self.d_model)

        self.encoder = Encoder(
            self.d_model,
            self.n_heads,
            self.d_ff,
            self.enc_layers,
            dropout=self.dropout,
            distill=self.distill,
            factor=self.factor,
            norm_mode=self.norm_mode,
            attention_type=self.attention_type,
        )

        self.proj = nn.Linear(self.d_model, 1)

    def forward(self, seq: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        L_total = seq.size(1)
        if L_total < self.enc_len + self.pred_len:
            raise ValueError(f"Input too short ({L_total}); need >= {self.enc_len + self.pred_len}")

        enc_x = seq[:, :self.enc_len, :]
        enc_h = self.pos_enc(self.enc_embedding(enc_x))
        if self.use_time_embedding and timestamps is not None:
            enc_ts = timestamps[:, :self.enc_len]
            enc_h = self.time_add_norm(enc_h + self.time_embedding(enc_ts))

        enc_out = self.encoder(enc_h)
        last_h = enc_out[:, -1, :]
        out = self.proj(last_h)
        return out.squeeze(-1).unsqueeze(-1)


# =====================================================
# Training helpers specific to encoder-only target types
# =====================================================

def build_target(batch_data: torch.Tensor, asset_index: int, target_type: str) -> torch.Tensor:
    """
    batch_data: [B, enc_len + 1, d_input], pred_len must be 1
    Returns: [B, 1]
    """
    if batch_data.size(1) < 2:
        raise ValueError("Need at least 2 time steps to compute change target.")
    y = batch_data[:, -1, asset_index]
    return y.unsqueeze(-1)
    # elif target_type == "change":
    #     return build_change_target_from_levels(batch_data, asset_index, pred_len=1)
    # else:
    #     raise ValueError("target_type must be 'price' or 'change'")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    asset_index: int = 0,
    log_interval: int = 10,
    val_interval: int = 100,
):
    def _eo_target_fn(batch_data: torch.Tensor, model: nn.Module, asset_index: int):
        return build_target(batch_data, asset_index, getattr(model, "target_type", "change"))

    return train_model_generic(
        model,
        train_loader,
        val_loader,
        cfg,
        target_fn=_eo_target_fn,
        target_kwargs={"asset_index": asset_index},
        logger=log,
        log_interval=log_interval,
        val_interval=val_interval,
    )
