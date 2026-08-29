"""
Fourier Neural Operator (FNO) for Irregular Mesh Flow Field Prediction

Based on: https://github.com/thuml/Neural-Solver-Library
Paper: Fourier Neural Operator for Parametric Partial Differential Equations (ICLR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .Basic import MLP, timestep_embedding
from .FNO_Layers import (
    IPHI,
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv2d_IrregularGeo,
    SpectralConv3d,
)

BlockList = [None, SpectralConv1d, SpectralConv2d, SpectralConv3d]
ConvList = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]


class FNO(nn.Module):
    """
    Fourier Neural Operator for irregular mesh PDE solving.

    Input:
        x: [B, N, space_dim] - coordinates
        fx: [B, N, fun_dim] - flow features

    Output:
        out: [B, N, out_dim] - predicted flow
    """
    def __init__(
        self,
        space_dim=2,
        fun_dim=4,
        out_dim=4,
        n_hidden=64,
        n_layers=4,
        modes=12,
        time_input=False,
        geotype='unstructured',
        shapelist=None,
        padding=0,
    ):
        super().__init__()
        self.__name__ = 'FNO'
        self.space_dim = space_dim
        self.fun_dim = fun_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.modes = modes
        self.time_input = time_input
        self.geotype = geotype
        self.shapelist = shapelist or [32, 32]  # Default grid size for structured
        self.padding = padding

        ## embedding
        self.preprocess = MLP(
            fun_dim + space_dim, n_hidden * 2, n_hidden,
            n_layers=0, res=False, act='gelu'
        )

        if time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(),
                nn.Linear(n_hidden, n_hidden)
            )

        # geometry projection
        if self.geotype == 'unstructured':
            self.fftproject_in = SpectralConv2d_IrregularGeo(
                n_hidden, n_hidden, modes, modes, self.shapelist[0], self.shapelist[1]
            )
            self.fftproject_out = SpectralConv2d_IrregularGeo(
                n_hidden, n_hidden, modes, modes, self.shapelist[0], self.shapelist[1]
            )
            self.iphi = IPHI()
            self.padding_list = [
                (16 - size % 16) % 16 for size in self.shapelist
            ]
        else:
            self.padding_list = [(16 - size % 16) % 16 for size in self.shapelist]

        # Fourier layers
        self.conv_layers = nn.ModuleList([
            BlockList[len(self.padding_list)](n_hidden, n_hidden, *[modes for _ in range(len(self.padding_list))])
            for _ in range(n_layers)
        ])
        self.w_layers = nn.ModuleList([
            ConvList[len(self.padding_list)](n_hidden, n_hidden, 1)
            for _ in range(n_layers)
        ])

        # projectors
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
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

        original_pos = x

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)

        if T is not None and self.time_input:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        if self.geotype == 'unstructured':
            x = self.fftproject_in(fx.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None)
        else:
            B, N, _ = x.shape
            x = fx.permute(0, 2, 1).reshape(B, self.n_hidden, *self.shapelist)
            if not all(item == 0 for item in self.padding_list):
                if len(self.shapelist) == 2:
                    x = F.pad(x, [0, self.padding_list[1], 0, self.padding_list[0]])
                elif len(self.shapelist) == 3:
                    x = F.pad(x, [0, self.padding_list[2], 0, self.padding_list[1], 0, self.padding_list[0]])

        for i in range(self.n_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            x = F.gelu(x)

        if self.geotype == 'unstructured':
            x = self.fftproject_out(x, x_out=original_pos, iphi=self.iphi, code=None).permute(0, 2, 1)
        else:
            if not all(item == 0 for item in self.padding_list):
                if len(self.shapelist) == 2:
                    x = x[..., :-self.padding_list[0], :-self.padding_list[1]]
                elif len(self.shapelist) == 3:
                    x = x[..., :-self.padding_list[0], :-self.padding_list[1], :-self.padding_list[2]]
            B = x.shape[0]
            x = x.reshape(B, self.n_hidden, -1).permute(0, 2, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x


class FNOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        return self.mse(pred, target).mean()


if __name__ == '__main__':
    B, N = 2, 1000

    model = FNO(
        space_dim=2,
        fun_dim=4,
        out_dim=4,
        n_hidden=64,
        n_layers=4,
        modes=12,
        geotype='unstructured',
        shapelist=[32, 32],
    )

    x = torch.randn(B, N, 2)  # coordinates
    fx = torch.randn(B, N, 4)  # flow features

    out = model(x, fx)

    print(f"Input x shape: {x.shape}")
    print(f"Input fx shape: {fx.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
