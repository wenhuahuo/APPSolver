"""Lightweight encoders for normalized numeric condition parameters."""

import math

import torch
import torch.nn as nn


class MLPConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierMLPConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_bands: int = 4):
        super().__init__()
        self.register_buffer(
            'frequencies',
            math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32)),
        )
        feature_dim = input_dim * (1 + 2 * num_bands)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x.unsqueeze(-1) * self.frequencies
        features = torch.cat(
            [x, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)],
            dim=-1,
        )
        return self.net(features)


class FiLMConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, x: torch.Tensor):
        gamma, beta = self.net(x).chunk(2, dim=-1)
        return gamma, beta


def build_condition_encoder(kind: str, input_dim: int, d_model: int):
    if kind == 'mlp':
        return MLPConditionEncoder(input_dim, d_model)
    if kind == 'fourier_mlp':
        return FourierMLPConditionEncoder(input_dim, d_model)
    if kind == 'film':
        return FiLMConditionEncoder(input_dim, d_model)
    raise ValueError(f"Unknown lightweight condition encoder: {kind}")
