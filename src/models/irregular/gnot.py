"""
General Neural Operator Transformer (GNOT) for Irregular Mesh Flow Field Prediction

Based on: https://github.com/thuml/Neural-Solver-Library
Paper: GNOT: A General Neural Operator Transformer for Operator Learning (ICML 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .Basic import ACTIVATION, MLP, LinearAttention, timestep_embedding


class GNOT_block(nn.Module):
    """Transformer encoder block in MOE style."""

    def __init__(self, num_heads: int, hidden_dim: int, dropout: float,
                 act='gelu', mlp_ratio=4, space_dim=2, n_experts=3):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)

        self.selfattn = LinearAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout
        )
        self.crossattn = LinearAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout
        )
        self.resid_drop1 = nn.Dropout(dropout)
        self.resid_drop2 = nn.Dropout(dropout)

        ## MLP in MOE
        self.n_experts = n_experts
        if act in ACTIVATION:
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError(f"Activation {act} not supported")
        self.moe_mlp1 = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        ) for _ in range(self.n_experts)])

        self.moe_mlp2 = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        ) for _ in range(self.n_experts)])

        self.gatenet = nn.Sequential(
            nn.Linear(space_dim, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, self.n_experts)
        )

    def forward(self, x, y, pos):
        ## point-wise gate for moe
        gate_score = F.softmax(self.gatenet(pos), dim=-1).unsqueeze(2)
        ## cross attention between geo and physics observation
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln2(y)))
        ## moe mlp
        x_moe1 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)], dim=-1)
        x_moe1 = (gate_score * x_moe1).sum(dim=-1, keepdim=False)
        x = x + self.ln3(x_moe1)
        ## self attention among geo
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        ## moe mlp
        x_moe2 = torch.stack([self.moe_mlp2[i](x) for i in range(self.n_experts)], dim=-1)
        x_moe2 = (gate_score * x_moe2).sum(dim=-1, keepdim=False)
        x = x + self.ln5(x_moe2)
        return x


class GNOT(nn.Module):
    """
    General Neural Operator Transformer for irregular mesh PDE solving.

    Input:
        x: [B, N, space_dim] - coordinates
        fx: [B, N, fun_dim] - flow features (optional)
        T: [B] - time step (optional)

    Output:
        out: [B, N, out_dim] - predicted flow
    """
    def __init__(
        self,
        space_dim=2,
        fun_dim=4,
        out_dim=4,
        n_hidden=256,
        n_heads=8,
        n_layers=3,
        mlp_ratio=4,
        dropout=0.0,
        act='gelu',
        n_experts=3,
        time_input=False,
        unified_pos=False,
        ref=8,
    ):
        super().__init__()
        self.__name__ = 'GNOT'
        self.space_dim = space_dim
        self.fun_dim = fun_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.act = act
        self.n_experts = n_experts
        self.time_input = time_input
        self.unified_pos = unified_pos
        self.ref = ref

        ## embedding
        self.preprocess_x = MLP(
            space_dim, n_hidden * 2, n_hidden,
            n_layers=0, res=False, act=act
        )
        self.preprocess_z = MLP(
            fun_dim + space_dim, n_hidden * 2, n_hidden,
            n_layers=0, res=False, act=act
        )

        if time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(),
                nn.Linear(n_hidden, n_hidden)
            )

        ## models
        self.blocks = nn.ModuleList([
            GNOT_block(
                num_heads=n_heads,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                space_dim=space_dim,
                n_experts=n_experts
            )
            for _ in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

        # projectors
        self.fc1 = nn.Linear(n_hidden, n_hidden * 2)
        self.fc2 = nn.Linear(n_hidden * 2, out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x, fx=None, T=None):
        """
        Forward pass.

        Args:
            x: [B, N, space_dim] coordinates
            fx: [B, N, fun_dim] flow features
            T: [B] time step (optional)

        Returns:
            out: [B, N, out_dim] predicted flow
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if fx is not None and fx.dim() == 2:
            fx = fx.unsqueeze(0)

        pos = x

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess_z(fx)
        else:
            fx = self.preprocess_z(x)

        fx = fx + self.placeholder[None, None, :]
        x = self.preprocess_x(x)

        if T is not None and self.time_input:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(x, fx, pos)

        fx = self.fc1(fx)
        fx = F.gelu(fx)
        fx = self.fc2(fx)

        return fx


class GNOTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        return self.mse(pred, target).mean()


if __name__ == '__main__':
    B, N = 2, 1000

    model = GNOT(
        space_dim=2,
        fun_dim=4,
        out_dim=4,
        n_hidden=128,
        n_heads=4,
        n_layers=3,
        n_experts=3,
    )

    x = torch.randn(B, N, 2)  # coordinates
    fx = torch.randn(B, N, 4)  # flow features

    out = model(x, fx)

    print(f"Input x shape: {x.shape}")
    print(f"Input fx shape: {fx.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
