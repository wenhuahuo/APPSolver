"""
DPT (Dense Prediction Transformer) with LLM-based Parameter Encoder

Extends DPT to accept ship parameters as text and use an LLM encoder
to generate a parameter token that is concatenated with patch tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .dpt import (
    MultiHeadSelfAttention,
    TransformerBlock,
    ViTEncoder,
    Reassemble,
    ResidualConvUnit1D,
    FeatureFusionBlock1D,
    Scratch1D,
)


class PatchEmbeddingWithParams(nn.Module):
    """
    Patch embedding that also incorporates a parameter token.
    
    Input:
        x: (B, P, flattened_dim) - patch features
        params_embed: (B, d_model) or (B, num_queries, d_model) - parameter embedding from LLM
    Output:
        (B, P + num_queries, d_model) if num_queries > 1, else (B, P + 1, d_model)
    """

    def __init__(
        self,
        flattened_dim: int,
        d_model: int,
        max_patches: int = 1024,
        dropout: float = 0.0,
        params_token_dim: int = 1,
    ):
        super().__init__()
        self.proj = nn.Linear(flattened_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.max_patches = max_patches
        self.params_token_dim = params_token_dim

    def forward(
        self,
        x: torch.Tensor,
        params_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, P, _ = x.shape
        x = self.proj(x)

        if P <= self.max_patches:
            pos = self.pos_emb[:, :P, :]
        else:
            pos = F.interpolate(
                self.pos_emb.permute(0, 2, 1),
                size=P,
                mode='linear',
                align_corners=False,
            ).permute(0, 2, 1)

        x = self.drop(x + pos)

        if params_embed is not None:
            if params_embed.dim() == 2:
                params_embed = params_embed.unsqueeze(1)
            x = torch.cat([params_embed, x], dim=1)

        return x


class DPTWithLLM(nn.Module):
    """
    DPT with LLM-based parameter encoder for ship parameters.

    Architecture:
        1. LLMEncoder (frozen): params_text -> (d_model) embedding
        2. PatchEmbeddingWithParams: patches + params_token -> (P+1, d_model)
        3. ViTEncoder with hooks for multi-scale features
        4. Reassemble blocks for multi-scale fusion
        5. Feature fusion and output projection

    Args:
        flattened_dim: N_points × C_channels per patch
        features: Feature width in fusion decoder
        d_model: Transformer hidden dimension
        llm_model_name: HuggingFace model name for LLM encoder
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
        use_bn: Use BatchNorm in residual conv units
        hooks: Block indices for intermediate feature extraction
        max_patches: Maximum number of patches for positional embedding
    """

    def __init__(
        self,
        flattened_dim: int,
        features: int = 256,
        d_model: int = 256,
        llm_model_name: str = 'Qwen/Qwen2.5-0.5B',
        n_heads: int = 8,
        n_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_bn: bool = False,
        hooks: Optional[List[int]] = None,
        max_patches: int = 1024,
    ):
        super().__init__()
        self.flattened_dim = flattened_dim
        self.d_model = d_model

        from .llm_encoder import LLMEncoder
        self.llm_encoder = LLMEncoder(
            model_name=llm_model_name,
            d_model=d_model,
            num_queries=1,
        )

        self.patch_embed = PatchEmbeddingWithParams(
            flattened_dim,
            d_model,
            max_patches=max_patches,
            dropout=dropout,
        )

        self.encoder = ViTEncoder(
            d_model,
            n_heads,
            n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            hooks=hooks,
        )

        re_ch = features
        self.reassemble1 = Reassemble(d_model, re_ch, stride=4, use_transpose=False)
        self.reassemble2 = Reassemble(d_model, re_ch, stride=2, use_transpose=False)
        self.reassemble3 = Reassemble(d_model, re_ch, stride=1, use_transpose=False)
        self.reassemble4 = Reassemble(d_model, re_ch, stride=1, use_transpose=True)

        self.scratch = Scratch1D(
            in_shapes=[re_ch, re_ch, re_ch, re_ch],
            features=features,
            out_dim=flattened_dim,
            use_bn=use_bn,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _align_sequence(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.shape[1] == target_len:
            return x
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x.permute(0, 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        params_text: Optional[List[str]] = None,
        mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, P, flattened_dim) - input patch features
            params_text: List[str] - ship parameters as text descriptions
            mask: (B, P, N) - optional validity mask
            device: torch.device - device for LLM encoding

        Returns:
            (B, P, flattened_dim) - predicted output patch features
        """
        B, P, _ = x.shape

        params_embed = None
        if params_text is not None and self.llm_encoder is not None:
            if device is None:
                device = x.device
            params_embeds = []
            for text in params_text:
                embed = self.llm_encoder(text, device)
                params_embeds.append(embed)
            params_embed = torch.stack(params_embeds, dim=0)

        tokens = self.patch_embed(x, params_embed)

        P_actual = tokens.shape[1]

        feat1, feat2, feat3, feat4 = self.encoder(tokens)

        if P_actual != P + 1:
            if P_actual > P + 1:
                tokens = tokens[:, :P+1, :]
                feat1 = feat1[:, :P+1, :]
                feat2 = feat2[:, :P+1, :]
                feat3 = feat3[:, :P+1, :]
                feat4 = feat4[:, :P+1, :]

        r1 = self.reassemble1(feat1)
        r2 = self.reassemble2(feat2)
        r3 = self.reassemble3(feat3)
        r4 = self.reassemble4(feat4)

        l1 = self.scratch.layer1_rn(r1)
        l2 = self.scratch.layer2_rn(r2)
        l3 = self.scratch.layer3_rn(r3)
        l4 = self.scratch.layer4_rn(r4)

        path4 = self.scratch.refinenet4(l4)

        path4_aligned = self._align_sequence(path4, l3.shape[1])
        path3 = self.scratch.refinenet3(l3, path4_aligned)

        path3_aligned = self._align_sequence(path3, l2.shape[1])
        path2 = self.scratch.refinenet2(l2, path3_aligned)

        path2_aligned = self._align_sequence(path2, l1.shape[1])
        path1 = self.scratch.refinenet1(l1, path2_aligned)

        out = self._align_sequence(path1, P_actual)

        out = self.scratch.output_conv(out)

        if out.shape[1] > P:
            out = out[:, :P, :]

        return out

    def get_trainable_params(self):
        """Return only trainable parameters (projection layer, not LLM)"""
        return list(self.patch_embed.parameters()) + \
               list(self.encoder.parameters()) + \
               list(self.reassemble1.parameters()) + \
               list(self.reassemble2.parameters()) + \
               list(self.reassemble3.parameters()) + \
               list(self.reassemble4.parameters()) + \
               list(self.scratch.parameters()) + \
               list(self.llm_encoder.get_trainable_params())

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override to only save trainable weights, not the frozen LLM"""
        state = {}
        state['patch_embed'] = self.patch_embed.state_dict(destination, prefix + 'patch_embed.', keep_vars)
        state['encoder'] = self.encoder.state_dict(destination, prefix + 'encoder.', keep_vars)
        state['reassemble1'] = self.reassemble1.state_dict(destination, prefix + 'reassemble1.', keep_vars)
        state['reassemble2'] = self.reassemble2.state_dict(destination, prefix + 'reassemble2.', keep_vars)
        state['reassemble3'] = self.reassemble3.state_dict(destination, prefix + 'reassemble3.', keep_vars)
        state['reassemble4'] = self.reassemble4.state_dict(destination, prefix + 'reassemble4.', keep_vars)
        state['scratch'] = self.scratch.state_dict(destination, prefix + 'scratch.', keep_vars)
        state['llm_encoder'] = self.llm_encoder.state_dict(destination, prefix + 'llm_encoder.', keep_vars)
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Override to load only trainable weights"""
        if 'patch_embed' in state_dict:
            self.patch_embed.load_state_dict(state_dict['patch_embed'], strict=strict)
        if 'encoder' in state_dict:
            self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
        if 'reassemble1' in state_dict:
            self.reassemble1.load_state_dict(state_dict['reassemble1'], strict=strict)
        if 'reassemble2' in state_dict:
            self.reassemble2.load_state_dict(state_dict['reassemble2'], strict=strict)
        if 'reassemble3' in state_dict:
            self.reassemble3.load_state_dict(state_dict['reassemble3'], strict=strict)
        if 'reassemble4' in state_dict:
            self.reassemble4.load_state_dict(state_dict['reassemble4'], strict=strict)
        if 'scratch' in state_dict:
            self.scratch.load_state_dict(state_dict['scratch'], strict=strict)
        if 'llm_encoder' in state_dict:
            self.llm_encoder.load_state_dict(state_dict['llm_encoder'], strict=strict)
