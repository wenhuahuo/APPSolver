"""
Simple Transformer for patch-based flow field prediction.

Input shape: [B, P, N*C_in] where:
- B: batch size
- P: number of patches
- N*C: flattened features per patch (N points × C channels)

Optional: params_embed [B, params_dim] is projected to d_model and prepended
to the sequence.

Output shape: [B, P, N*C_out] - predicted flow at next timestep
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition_encoders import build_condition_encoder


class Transformer(nn.Module):
    """
    Unified Transformer for patch-based flow field prediction.
    
    Supports optional parameter embedding:
    - Without embedding: [B, P, in_flattened_dim] -> [B, P, out_flattened_dim]
    - With embedding: [B, P, in_flattened_dim] + [B, params_dim]
      -> [B, P, out_flattened_dim]. The parameter embedding is projected to
      d_model, prepended to the sequence, then removed from output.
    """
    
    def __init__(
        self,
        in_flattened_dim: int,
        out_flattened_dim: int = None,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_patches: int = 10000,
        params_dim: int = None,
        condition_encoder: str = 'token',
    ):
        super().__init__()
        
        if out_flattened_dim is None:
            out_flattened_dim = in_flattened_dim

        self.in_flattened_dim = in_flattened_dim
        self.out_flattened_dim = out_flattened_dim
        self.d_model = d_model
        self.params_dim = params_dim
        self.condition_encoder_type = condition_encoder
        
        self.input_proj = nn.Linear(in_flattened_dim, d_model)
        self.params_proj = None
        self.condition_module = None
        if params_dim is not None:
            if condition_encoder == 'token':
                self.params_proj = (
                    nn.Identity() if params_dim == d_model
                    else nn.Linear(params_dim, d_model)
                )
            else:
                self.condition_module = build_condition_encoder(
                    condition_encoder, params_dim, d_model
                )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, out_flattened_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        params_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, P, in_flattened_dim) - input patch features
            params_embed: (B, params_dim) or None - optional pre-computed LLM embedding
        
        Returns:
            (B, P, out_flattened_dim) - predicted flow features
        """
        B, P, _ = x.shape
        
        x_proj = self.input_proj(x)
        
        if P <= self.pos_embed.size(1):
            pos = self.pos_embed[:, :P, :]
        else:
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=P,
                mode='linear',
                align_corners=False,
            ).permute(0, 2, 1)
        
        x_proj = x_proj + pos
        x_proj = self.drop(x_proj)
        
        has_condition_token = False
        if params_embed is not None:
            if params_embed.dim() == 1:
                params_embed = params_embed.unsqueeze(0)
            if self.params_dim is None or params_embed.shape[-1] != self.params_dim:
                raise ValueError(
                    f"params_embed dim {params_embed.shape[-1]} does not match "
                    f"configured params_dim {self.params_dim}."
                )

            if self.condition_encoder_type == 'film':
                gamma, beta = self.condition_module(params_embed)
                x_proj = x_proj * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            else:
                condition_token = (
                    self.params_proj(params_embed)
                    if self.condition_encoder_type == 'token'
                    else self.condition_module(params_embed)
                )
                x_proj = torch.cat([condition_token.unsqueeze(1), x_proj], dim=1)
                has_condition_token = True
        
        x_trans = self.transformer(x_proj)
        
        if has_condition_token:
            x_trans = x_trans[:, 1:, :]
        
        x_out = self.output_proj(x_trans)
        
        return x_out


class LearnedSliceAttention(nn.Module):
    """Transolver-style soft assignment from patches to learned slices."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        slice_num: int,
    ):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.temperature = nn.Parameter(torch.full((1, nhead, 1, 1), 0.5))
        self.feature_proj = nn.Linear(d_model, d_model)
        self.assignment_proj = nn.Linear(d_model, d_model)
        self.slice_proj = nn.Linear(self.head_dim, slice_num)
        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, patch_count, width = x.shape
        features = self.feature_proj(x).reshape(
            batch_size, patch_count, self.nhead, self.head_dim
        ).permute(0, 2, 1, 3)
        assignment_features = self.assignment_proj(x).reshape(
            batch_size, patch_count, self.nhead, self.head_dim
        ).permute(0, 2, 1, 3)
        assignment = torch.softmax(
            self.slice_proj(assignment_features) / self.temperature,
            dim=-1,
        )
        mass = assignment.sum(dim=2)
        slices = torch.einsum('bhpd,bhps->bhsd', features, assignment)
        slices = slices / mass.clamp_min(1e-5).unsqueeze(-1)

        attention = torch.softmax(
            torch.matmul(self.to_q(slices), self.to_k(slices).transpose(-1, -2))
            * self.scale,
            dim=-1,
        )
        slices = torch.matmul(self.dropout(attention), self.to_v(slices))
        output = torch.einsum('bhsd,bhps->bhpd', slices, assignment)
        output = output.permute(0, 2, 1, 3).reshape(
            batch_size, patch_count, width
        )
        return self.out_proj(output)


class LearnedPatchBlock(nn.Module):
    """Transolver-style learned slicing over patch embeddings."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        slice_num: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = LearnedSliceAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feedforward(self.norm2(x))


class LearnedPatchTransformer(Transformer):
    """Patch Transformer with learned patch-to-slice assignment in each block."""

    def __init__(
        self,
        in_flattened_dim: int,
        out_flattened_dim: int = None,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_patches: int = 10000,
        params_dim: int = None,
        condition_encoder: str = 'token',
        slice_num: int = 32,
    ):
        super().__init__(
            in_flattened_dim=in_flattened_dim,
            out_flattened_dim=out_flattened_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_patches=max_patches,
            params_dim=params_dim,
            condition_encoder=condition_encoder,
        )
        self.slice_num = slice_num
        self.transformer = nn.Sequential(*[
            LearnedPatchBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                slice_num=slice_num,
            )
            for _ in range(num_layers)
        ])
        self._init_weights()


class TransformerLoss(nn.Module):
    """MSE loss with optional mask support."""
    
    def __init__(self, use_mask: bool = True):
        super().__init__()
        self.use_mask = use_mask
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        loss = self.mse(pred, target)
        
        if self.use_mask and mask is not None:
            mask_float = mask.float()
            if mask_float.dim() == 2:
                mask_float = mask_float.unsqueeze(-1).expand_as(loss)
            elif mask_float.dim() == 3:
                B, P, N = mask_float.shape
                C = loss.size(-1) // N
                mask_float = mask_float.unsqueeze(-1).expand(B, P, N, C)
                mask_float = mask_float.reshape(B, P, N * C)
            loss = loss * mask_float
            return loss.sum() / mask_float.sum().clamp_min(1.0)
        
        return loss.mean()


if __name__ == '__main__':
    B, P, NC = 4, 100, 96
    in_flattened_dim = NC
    out_flattened_dim = NC // 2
    
    model = Transformer(
        in_flattened_dim=in_flattened_dim,
        out_flattened_dim=out_flattened_dim,
        d_model=128,
        nhead=4,
        num_layers=3,
    )
    
    x = torch.randn(B, P, NC)
    pred = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    embed = torch.randn(B, 128)
    pred_with_embed = model(x, params_embed=embed)
    print(f"With embedding - Output shape: {pred_with_embed.shape}")
