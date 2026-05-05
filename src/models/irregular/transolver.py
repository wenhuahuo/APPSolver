"""
Transolver for Irregular Mesh Flow Field Prediction

Integrated from Transolver_Irregular_Mesh.py and Physics_Attention.py
Input shape: [B, N, C] where:
- B: batch size
- N: number of points
- C: channels (coord + flow features)

Output shape: [B, N, C] - predicted flow at next timestep
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class PhysicsAttentionIrregularMesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
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
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class TransolverBlock(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = PhysicsAttentionIrregularMesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver(nn.Module):
    def __init__(
        self,
        space_dim=2,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act='gelu',
        mlp_ratio=1,
        fun_dim=4,
        out_dim=4,
        slice_num=32,
        ref=8,
        unified_pos=False
    ):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                last_layer=(_ == n_layers - 1)
            )
            for _ in range(n_layers)
        ])
        
        self._initialize_weights()
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

    def _initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        self.apply(_init_weights)

    def get_grid(self, x, batchsize=1):
        gridx = torch.linspace(0, 1, self.ref, dtype=torch.float, device=x.device)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.linspace(0, 1, self.ref, dtype=torch.float, device=x.device)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).reshape(batchsize, self.ref * self.ref, 2)

        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1))
        pos = pos.reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx):
        if self.unified_pos:
            x = self.get_grid(x, x.shape[0])
        
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        return fx


class TransolverLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        return self.mse(pred, target).mean()


if __name__ == '__main__':
    B, N, C = 2, 1000, 6
    
    model = Transolver(
        space_dim=2,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act='gelu',
        mlp_ratio=1,
        fun_dim=4,
        out_dim=4,
        slice_num=32,
        ref=8,
        unified_pos=False
    )
    
    pos = torch.randn(B, N, 2)
    flow = torch.randn(B, N, 4)
    
    out = model(pos, flow)
    
    print(f"Input pos shape: {pos.shape}")
    print(f"Input flow shape: {flow.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")