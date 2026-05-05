"""
Patch-based Models with Pre-computed Parameter Embeddings

These models accept pre-computed parameter embeddings directly,
without any LLM code. The embeddings are pre-computed using a separate
preprocessing step (see scripts/precompute_ship_embeddings.py).

Input shapes:
    - x: (B, P, N*C) - patch features
    - params_embed: (B, d_model) - pre-computed parameter embedding

Output shape: (B, P, N*C) - predicted flow at next timestep
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbeddingWithEmbedding(nn.Module):
    """
    Patch embedding that incorporates a pre-computed parameter embedding.
    No LLM involved.
    """
    def __init__(self, flattened_dim: int, d_model: int, dropout: float = 0.1, max_patches: int = 10000):
        super().__init__()
        self.input_proj = nn.Linear(flattened_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.max_patches = max_patches
    
    def forward(self, x: torch.Tensor, params_embed: torch.Tensor = None) -> torch.Tensor:
        B, P, _ = x.shape
        
        x_proj = self.input_proj(x)
        
        if P <= self.max_patches:
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
            if params_embed.dim() == 2:
                params_embed = params_embed.unsqueeze(1)
            x_proj = torch.cat([params_embed, x_proj], dim=1)
        
        return x_proj


class PatchTransformerWithEmbedding(nn.Module):
    """
    Simple Transformer with pre-computed parameter embedding.
    No LLM code - embeddings must be pre-computed.
    """
    def __init__(
        self,
        flattened_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.flattened_dim = flattened_dim
        self.d_model = d_model
        
        self.patch_embed = PatchEmbeddingWithEmbedding(flattened_dim, d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, flattened_dim)
        
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
        B, P, NC = x.shape
        
        x_proj = self.patch_embed(x, params_embed)
        
        P_actual = x_proj.shape[1]
        
        x_trans = self.transformer(x_proj)
        
        if P_actual > P:
            x_trans = x_trans[:, :P, :]
        
        x_out = self.output_proj(x_trans)
        
        return x_out


class DPTWithEmbedding(nn.Module):
    """
    DPT with pre-computed parameter embedding.
    No LLM code - embeddings must be pre-computed.
    """
    
    def __init__(
        self,
        flattened_dim: int,
        features: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_bn: bool = False,
        max_patches: int = 1024,
    ):
        super().__init__()
        self.flattened_dim = flattened_dim
        self.d_model = d_model
        
        self.patch_embed = PatchEmbeddingWithEmbedding(
            flattened_dim, d_model, dropout, max_patches=max_patches
        )
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        
        re_ch = features
        self.reassemble1 = self._make_reassemble(d_model, re_ch, stride=4)
        self.reassemble2 = self._make_reassemble(d_model, re_ch, stride=2)
        self.reassemble3 = self._make_reassemble(d_model, re_ch, stride=1)
        self.reassemble4 = self._make_reassemble(d_model, re_ch, stride=1, use_transpose=True)
        
        self.scratch = self._make_scratch(re_ch, features)
        
        self._init_weights()
    
    def _make_reassemble(self, d_model, out_channels, stride=1, use_transpose=False):
        class ReassembleBlock(nn.Module):
            def __init__(self, d_model, out_channels, stride, use_transpose):
                super().__init__()
                if use_transpose:
                    self.conv = nn.ConvTranspose1d(d_model, out_channels, kernel_size=2, stride=2)
                else:
                    padding = 1 if stride == 1 else 0
                    self.conv = nn.Conv1d(d_model, out_channels, kernel_size=stride if stride > 1 else 1, stride=max(stride, 1), padding=padding)
                self.norm = nn.LayerNorm(out_channels)
            
            def forward(self, x):
                x = x.permute(0, 2, 1)
                x = self.conv(x)
                x = x.permute(0, 2, 1)
                return self.norm(x)
        
        return ReassembleBlock(d_model, out_channels, stride, use_transpose)
    
    def _make_scratch(self, re_ch, features):
        class Scratch(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1_rn = nn.Linear(re_ch, features)
                self.layer2_rn = nn.Linear(re_ch, features)
                self.layer3_rn = nn.Linear(re_ch, features)
                self.layer4_rn = nn.Linear(re_ch, features)
                self.refinenet4 = self._make_refinenet(features)
                self.refinenet3 = self._make_refinenet(features)
                self.refinenet2 = self._make_refinenet(features)
                self.refinenet1 = self._make_refinenet(features)
                self.output_conv = nn.Sequential(
                    nn.Linear(features, features // 2),
                    nn.GELU(),
                    nn.Linear(features // 2, 64),
                )
            
            def _make_refinenet(self, features):
                class RefineNet(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.rcu1 = self._make_rcu(features)
                        self.rcu2 = self._make_rcu(features)
                        self.out_proj = nn.Linear(features, features)
                    
                    def _make_rcu(self, features):
                        return nn.Sequential(
                            nn.Conv1d(features, features, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv1d(features, features, kernel_size=3, padding=1),
                        )
                    
                    def forward(self, x, skip=None):
                        out = x
                        if skip is not None:
                            out = out + self.rcu1(skip)
                        out = self.rcu2(out)
                        out = F.interpolate(out.unsqueeze(-1), scale_factor=2, mode='linear', align_corners=False).squeeze(-1)
                        return self.out_proj(out.permute(0, 2, 1)).permute(0, 2, 1)
                
                return RefineNet()
            
            def forward(self, r1, r2, r3, r4):
                l1 = self.layer1_rn(r1)
                l2 = self.layer2_rn(r2)
                l3 = self.layer3_rn(r3)
                l4 = self.layer4_rn(r4)
                
                path4 = self.refinenet4(l4)
                path3 = self.refinenet3(l3, path4)
                path2 = self.refinenet2(l2, path3)
                path1 = self.refinenet1(l1, path2)
                
                return self.output_conv(path1)
        
        return Scratch()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _align_sequence(self, x, target_len):
        if x.shape[1] == target_len:
            return x
        return F.interpolate(x.permute(0, 2, 1), size=target_len, mode='linear', align_corners=False).permute(0, 2, 1)
    
    def forward(self, x, params_embed=None):
        B, P, _ = x.shape
        
        tokens = self.patch_embed(x, params_embed)
        tokens = self.encoder(tokens)
        
        r1 = self.reassemble1(tokens)
        r2 = self.reassemble2(tokens)
        r3 = self.reassemble3(tokens)
        r4 = self.reassemble4(tokens)
        
        out = self.scratch(r1, r2, r3, r4)
        
        if out.shape[1] > P:
            out = out[:, :P, :]
        
        return out
