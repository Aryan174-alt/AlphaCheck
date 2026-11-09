"""A3 AI Lab: Deep Sequence Models (PyTorch Transformer Encoder & TCN).

Operates on rolling temporal sequences of features (e.g. past 20 trading days x num_features).
Evaluated on identical walk-forward splits as gradient boosting (RQ6).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, : x.size(1)]


class SmallTransformerClassifier(nn.Module):
    """Small PyTorch Transformer Encoder for multi-variate price-feature sequences."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        proj = self.input_proj(x)
        pos = self.pos_encoder(proj)
        out = self.transformer_encoder(pos)
        # Global average pooling over time steps
        pooled = out.mean(dim=1)
        logits = self.fc(pooled).squeeze(-1)
        return logits


class SmallTCNClassifier(nn.Module):
    """Small Temporal Convolutional Network (TCN) with causal dilated 1D convolutions."""

    def __init__(self, input_dim: int, num_channels: list[int] = [32, 32], kernel_size: int = 3):
        super().__init__()
        layers = []
        in_c = input_dim
        for i, out_c in enumerate(num_channels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            conv = nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation)
            layers.extend([conv, nn.ReLU(), nn.Dropout(0.1)])
            in_c = out_c

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(in_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim] -> conv requires [batch, input_dim, seq_len]
        x_trans = x.transpose(1, 2)
        out = self.network(x_trans)
        # Take the last time step output (causal)
        last_step = out[:, :, -1]
        logits = self.fc(last_step).squeeze(-1)
        return logits

