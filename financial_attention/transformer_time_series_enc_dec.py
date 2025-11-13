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

Notes:
- This refactored wrapper imports core components from financial_attention.components
  to avoid duplication and keep model behavior and documentation consistent.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging

from components import (
    Encoder,
    Decoder,
    PositionalEncoding,
    TimeEmbedding,
    TrainConfig,
    train_model_generic,
    build_change_target_from_levels,
)


logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger("informer_forecaster")


class InformerForecaster(nn.Module):
    def __init__(self, config, asset_index: int = 0):
        super().__init__()

        # Config
        self.asset_index = asset_index
        self.d_input = config["d_input"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_ff = config["d_ff"]
        self.enc_layers = config["enc_layers"]
        self.dec_layers = config["dec_layers"]
        self.dropout = config["dropout"]
        self.distill = config["distill"]
        self.factor = config.get("factor", 5)
        self.norm_mode = config.get("norm_mode", "pre")
        self.use_time_embedding = config.get("use_time_embedding", False)
        self.target_type = config.get("target_type", "price")
        if self.target_type not in ("price", "change"):
            raise ValueError("target_type must be 'price' or 'change'")

        # Sequence lengths
        self.enc_len = config["enc_len"]
        self.pred_len = config["pred_len"]
        self.guiding_len = config.get("guiding_len", 16)

        # Embeddings
        self.enc_embedding = nn.Linear(self.d_input, self.d_model)
        self.dec_embedding = nn.Linear(self.d_input, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model)
        if self.use_time_embedding:
            self.time_embedding = TimeEmbedding(self.d_model, embed_dim=8, dropout=self.dropout)
            self.time_add_norm = nn.LayerNorm(self.d_model)

        # Stacks
        self.encoder = Encoder(
            self.d_model,
            self.n_heads,
            self.d_ff,
            self.enc_layers,
            dropout=self.dropout,
            distill=self.distill,
            factor=self.factor,
            norm_mode=self.norm_mode,
            attention_type="prob",
        )
        self.decoder = Decoder(
            self.d_model,
            self.n_heads,
            self.d_ff,
            self.dec_layers,
            dropout=self.dropout,
            factor=self.factor,
            norm_mode=self.norm_mode,
        )

        # Projection to 1D target
        self.proj = nn.Linear(self.d_model, 1)

    def forward(self, seq: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        # Encoder
        enc_x = seq[:, :self.enc_len, :]
        enc_h = self.pos_enc(self.enc_embedding(enc_x))
        if self.use_time_embedding and timestamps is not None:
            enc_ts = timestamps[:, :self.enc_len]
            enc_h = self.time_add_norm(enc_h + self.time_embedding(enc_ts))
        enc_out = self.encoder(enc_h)

        # Decoder inputs
        dec_context = seq[:, self.enc_len - self.guiding_len: self.enc_len, :]
        dec_future = seq[:, self.enc_len: self.enc_len + self.pred_len, :].clone()
        dec_future[:, :, self.asset_index] = 0.0  # mask target
        dec_input = torch.cat([dec_context, dec_future], dim=1)
        dec_h = self.pos_enc(self.dec_embedding(dec_input))
        if self.use_time_embedding and timestamps is not None:
            dec_ts = timestamps[:, self.enc_len - self.guiding_len: self.enc_len + self.pred_len]
            dec_h = self.time_add_norm(dec_h + self.time_embedding(dec_ts))

        dec_out = self.decoder(dec_h, enc_out)
        out = self.proj(dec_out[:, -self.pred_len:, :])
        return out.squeeze(-1)


# =====================================================
# Training loop (shared optimizer/LR utilities imported from components)
# =====================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    asset_index: int = 0,
    log_interval: int = 10,
    val_interval: int = 100,
):
    def _dec_target_fn(batch_data: torch.Tensor, model: nn.Module, asset_index: int):
        pred_len = getattr(model, "pred_len", batch_data.size(1))
        target_type = getattr(model, "target_type", "price")
        if target_type == "price":
            return batch_data[:, -pred_len:, asset_index]
        return build_change_target_from_levels(batch_data, asset_index, pred_len=pred_len)

    return train_model_generic(
        model,
        train_loader,
        val_loader,
        cfg,
        target_fn=_dec_target_fn,
        target_kwargs={"asset_index": asset_index},
        logger=log,
        log_interval=log_interval,
        val_interval=val_interval,
    )
