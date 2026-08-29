"""
Universal Physics Transformer (UPT) for Irregular Mesh Flow Field Prediction

Based on: https://github.com/thuml/Neural-Solver-Library
Paper: Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators (NeurIPS 2024)
"""

from functools import partial

import torch
import torch.nn as nn
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import (
    DitBlock,
    Mlp,
    PerceiverBlock,
    PerceiverPoolingBlock,
    PrenormBlock,
)

from .Basic import timestep_embedding

################################################################
# UPT Mesh Encoder
################################################################

class RansPerceiver_Encoder(nn.Module):
    def __init__(
            self,
            dim,
            num_attn_heads,
            num_output_tokens,
            add_type_token=False,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            input_shape: tuple | None = None,
            fun_dim=0,
            time_input=False,
            n_hidden=256,
    ):
        super().__init__()
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.num_output_tokens = num_output_tokens
        self.add_type_token = add_type_token
        self.input_shape = input_shape
        assert self.input_shape is not None
        self.fun_dim = fun_dim
        self.time_input = time_input

        # set ndim
        _, ndim = self.input_shape
        ndim = ndim - self.fun_dim

        # pos_embed
        if self.fun_dim != 0:
            self.pos_embed = ContinuousSincosEmbed(dim=dim - self.fun_dim, ndim=ndim)
        else:
            self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)

        # perceiver
        self.mlp = Mlp(in_dim=dim, hidden_dim=dim * 4, init_weights=init_weights)
        self.block = PerceiverPoolingBlock(
            dim=dim,
            num_heads=num_attn_heads,
            num_query_tokens=num_output_tokens,
            perceiver_kwargs={
                'init_weights': init_weights,
                'init_last_proj_zero': init_last_proj_zero,
            },
        )

        if add_type_token:
            self.type_token = nn.Parameter(torch.empty(size=(1, 1, dim,)))
        else:
            self.type_token = None

        # output shape
        self.output_shape = (num_output_tokens, dim)

        if self.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(),
                nn.Linear(n_hidden, n_hidden)
            )

    def forward(self, x, fx=None, T=None):
        x = self.pos_embed(x)

        if self.fun_dim != 0 and fx is not None:
            x = torch.cat([x, fx], dim=-1)

        mask = None

        if T is not None and self.time_input:
            Time_emb = timestep_embedding(T, self.dim).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            x = x + Time_emb

        # perceiver
        x = self.mlp(x)
        x = self.block(kv=x, attn_mask=mask)

        if self.add_type_token:
            x = x + self.type_token

        return x


################################################################
# UPT Latent Transformer
################################################################

class TransformerModel(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_attn_heads,
            drop_path_rate=0.0,
            drop_path_decay=True,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            input_shape: tuple | None = None,
            condition_dim=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        self.input_shape = input_shape
        self.condition_dim = condition_dim

        assert self.input_shape is not None
        assert len(self.input_shape) == 2
        seqlen, input_dim = self.input_shape
        self.output_shape = (seqlen, dim)

        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # blocks
        if self.condition_dim is not None:
            block_ctor = partial(DitBlock, cond_dim=self.condition_dim)
        else:
            block_ctor = PrenormBlock
        if drop_path_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            block_ctor(
                dim=dim,
                num_heads=num_attn_heads,
                drop_path=dpr[i],
                init_weights=init_weights,
                init_last_proj_zero=init_last_proj_zero,
            )
            for i in range(self.depth)
        ])

    def forward(self, x, condition=None, static_tokens=None):
        assert x.ndim == 3

        # concat static tokens
        if static_tokens is not None:
            x = torch.cat([static_tokens, x], dim=1)

        # input projection
        x = self.input_proj(x)

        # apply blocks
        blk_kwargs = {'cond': condition} if condition is not None else {}
        for blk in self.blocks:
            x = blk(x, **blk_kwargs)

        # remove static tokens
        if static_tokens is not None:
            num_static_tokens = static_tokens.size(1)
            x = x[:, num_static_tokens:]

        return x


################################################################
# UPT Decoder
################################################################

class RansPerceiver_Decoder(nn.Module):
    def __init__(
            self,
            dim,
            num_attn_heads,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            use_last_norm=False,
            input_shape: tuple | None = None,
            ndim: int | None = None,
            output_shape: tuple | None = None,
            fun_dim=0,
            time_input=False,
            n_hidden=256,
    ):
        super().__init__()
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.use_last_norm = use_last_norm
        self.input_shape = input_shape
        assert ndim is not None
        self.ndim = ndim - fun_dim
        self.output_shape = output_shape
        self.fun_dim = fun_dim
        self.time_input = time_input

        # input projection
        assert self.input_shape is not None
        _, input_dim = self.input_shape
        self.proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # query tokens (create them from a positional embedding)
        if self.fun_dim != 0:
            self.pos_embed = ContinuousSincosEmbed(dim=dim - self.fun_dim, ndim=self.ndim)
        else:
            self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=self.ndim)
        self.query_mlp = Mlp(in_dim=dim, hidden_dim=dim, init_weights=init_weights)

        # latent to pixels
        self.perceiver = PerceiverBlock(
            dim=dim,
            num_heads=num_attn_heads,
            init_last_proj_zero=init_last_proj_zero,
            init_weights=init_weights,
        )
        assert self.output_shape is not None
        _, output_dim = self.output_shape
        self.norm = nn.LayerNorm(dim, eps=1e-6) if use_last_norm else nn.Identity()
        self.pred = LinearProjection(dim, output_dim, init_weights=init_weights)

        if self.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(),
                nn.Linear(n_hidden, n_hidden)
            )

    def forward(self, x, query_x, query_fx=None, T=None):
        # input projection
        x = self.proj(x)

        query_x = self.pos_embed(query_x)

        # create query
        if self.fun_dim != 0 and query_fx is not None:
            query_x = torch.cat([query_x, query_fx], dim=-1)
        query_x = self.query_mlp(query_x)

        if T is not None and self.time_input:
            Time_emb = timestep_embedding(T, self.dim).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            x = x + Time_emb

        # decode
        x = self.perceiver(q=query_x, kv=x)
        x = self.norm(x)
        x = self.pred(x)

        return x


################################################################
# UPT Main Model
################################################################

class UPT(nn.Module):
    """
    Universal Physics Transformer for irregular mesh PDE solving.

    Input:
        x: [B, N, space_dim] - coordinates
        fx: [B, N, fun_dim] - flow features (optional)
        T: [B] - time step (optional, for temporal tasks)

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
        num_output_tokens=32,
        drop_path_rate=0.3,
        time_input=False,
        geotype='unstructured',
        unified_pos=False,
        ref=8,
    ):
        super().__init__()
        self.__name__ = 'UPT'
        self.space_dim = space_dim
        self.fun_dim = fun_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_output_tokens = num_output_tokens
        self.time_input = time_input
        self.geotype = geotype
        self.unified_pos = unified_pos
        self.ref = ref

        self.input_shape = (None, space_dim + fun_dim)
        self.output_shape = (None, out_dim)

        # mesh_encoder
        self.mesh_encoder = RansPerceiver_Encoder(
            num_output_tokens=num_output_tokens,
            add_type_token=False,
            init_weights='truncnormal',
            dim=n_hidden,
            num_attn_heads=n_heads,
            input_shape=self.input_shape,
            fun_dim=fun_dim,
            time_input=time_input,
            n_hidden=n_hidden,
        )

        # latent
        self.latent = TransformerModel(
            init_weights='truncnormal',
            drop_path_rate=drop_path_rate,
            drop_path_decay=False,
            dim=n_hidden,
            num_attn_heads=n_heads,
            depth=n_layers,
            input_shape=self.mesh_encoder.output_shape,
            condition_dim=None
        )

        # decoder
        self.decoder = RansPerceiver_Decoder(
            init_weights='truncnormal',
            dim=n_hidden,
            num_attn_heads=n_heads,
            input_shape=self.latent.output_shape,
            ndim=self.input_shape[1],
            output_shape=self.output_shape,
            fun_dim=fun_dim,
            use_last_norm=True,
            time_input=time_input,
            n_hidden=n_hidden,
        )

    def forward(self, x, fx=None, T=None):
        """
        Forward pass for unstructured geometry.

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

        # encode
        embed = self.mesh_encoder(x=x, fx=fx, T=T)

        # propagate
        propagated = self.latent(embed)

        # decode
        x_hat = self.decoder(propagated, query_x=x, query_fx=fx, T=T)

        return x_hat


class UPTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        return self.mse(pred, target).mean()


if __name__ == '__main__':
    B, N = 2, 1000

    model = UPT(
        space_dim=2,
        fun_dim=4,
        out_dim=4,
        n_hidden=128,
        n_heads=4,
        n_layers=3,
        num_output_tokens=32,
    )

    x = torch.randn(B, N, 2)  # coordinates
    fx = torch.randn(B, N, 4)  # flow features

    out = model(x, fx)

    print(f"Input x shape: {x.shape}")
    print(f"Input fx shape: {fx.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
