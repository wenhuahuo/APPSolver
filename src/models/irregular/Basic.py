"""
Basic layers shared by the irregular-mesh models.
"""

import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

# Stores zero-arg factories: ACTIVATION[name]() returns a fresh activation module.
ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": partial(nn.LeakyReLU, 0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """Create sinusoidal timestep embeddings of shape [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :, :1])], dim=-1
        )
    return embedding


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()

        if act in ACTIVATION:
            act_fn = ACTIVATION[act]
        else:
            raise NotImplementedError(f"Activation {act} not supported")

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res

        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class LinearAttention(nn.Module):
    """
    Linear attention mechanism.
    """

    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, attn_type="l1", **kwargs
    ):
        super().__init__()
        self.k_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.n_head = heads
        self.dim_head = dim_head
        self.attn_type = attn_type

    def forward(self, x, y=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()

        q = self.q_proj(x).view(B, T1, self.n_head, self.dim_head).transpose(1, 2)
        k = self.k_proj(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)
        v = self.v_proj(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)

        if self.attn_type == "l1":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1.0 / (q * k_cumsum).sum(dim=-1, keepdim=True)
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            D_inv = 1.0 / T2
        elif self.attn_type == "l2":
            q = q / q.norm(dim=-1, keepdim=True, p=1)
            k = k / k.norm(dim=-1, keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1.0 / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        y = rearrange(y, "b h n d -> b n (h d)")
        y = self.proj(y)
        return y
