"""
Simple Transformer for Patch-based Flow Field Prediction

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


class PatchTransformer(nn.Module):
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
    ):
        super().__init__()
        
        if out_flattened_dim is None:
            out_flattened_dim = in_flattened_dim

        self.in_flattened_dim = in_flattened_dim
        self.out_flattened_dim = out_flattened_dim
        self.d_model = d_model
        self.params_dim = params_dim
        
        self.input_proj = nn.Linear(in_flattened_dim, d_model)
        if params_dim is None:
            self.params_proj = None
        elif params_dim == d_model:
            self.params_proj = nn.Identity()
        else:
            self.params_proj = nn.Linear(params_dim, d_model)
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
        
        if params_embed is not None:
            if params_embed.dim() == 1:
                params_embed = params_embed.unsqueeze(0)
            if self.params_proj is None:
                if params_embed.shape[-1] != self.d_model:
                    raise ValueError(
                        f"params_embed dim {params_embed.shape[-1]} does not match "
                        f"d_model {self.d_model}; initialize PatchTransformer with params_dim."
                    )
            else:
                if params_embed.shape[-1] != self.params_dim:
                    raise ValueError(
                        f"params_embed dim {params_embed.shape[-1]} does not match "
                        f"configured params_dim {self.params_dim}."
                    )
                params_embed = self.params_proj(params_embed)
            params_embed = params_embed.unsqueeze(1)
            x_proj = torch.cat([params_embed, x_proj], dim=1)
        
        x_trans = self.transformer(x_proj)
        
        if params_embed is not None:
            x_trans = x_trans[:, 1:, :]
        
        x_out = self.output_proj(x_trans)
        
        return x_out


class PatchTransformerLoss(nn.Module):
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
            denom = mask_float.sum()
            if denom > 0:
                return loss.sum() / denom
            return loss.mean()
        
        return loss.mean()


if __name__ == '__main__':
    B, P, NC = 4, 100, 96
    in_flattened_dim = NC
    out_flattened_dim = NC // 2
    
    model = PatchTransformer(
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
