"""
Dense Prediction Transformer (DPT) for Patch-based Flow Field Prediction

Adapted from: https://github.com/isl-org/DPT
Paper: "Vision Transformers for Dense Prediction" (Ranftl et al., ICCV 2021)

Key adaptation from the original DPT:
  Original DPT takes image tensors (B, C, H, W), applies a Conv2d patch embedding,
  feeds tokens through a ViT backbone, then reassembles 4 intermediate feature maps
  back into 2D spatial maps and fuses them with bilinear upsampling.

  This version takes pre-split patch tensors (B, P, N*C) directly, where:
    B = batch size
    P = number of patches (sequence length, treated as the "spatial" axis)
    N*C = flattened features per patch (N points × C channels per point)

  Differences:
  1. PatchEmbedding: Linear projection (B, P, N*C) → (B, P, d_model) instead of
     Conv2d on images.  A learnable 1-D positional embedding over P positions is
     added (no 2-D grid needed).
  2. ViT Encoder: Standard Transformer encoder (no timm dependency) with a
     configurable number of layers.  Intermediate activations are captured at
     4 equally-spaced hook depths.
  3. Reassemble blocks: Each hooked feature map (B, P, d_model) is projected to
     a different channel width and optionally down-sampled along the P axis
     (via strided Conv1d), mirroring the multi-scale hierarchy of the original.
  4. Feature Fusion (Refinement): Bottom-up fusion using 1-D residual conv units
     and linear interpolation along the P axis instead of 2-D bilinear upsampling.
  5. Output head: Linear projection back to the original (B, P, N*C) shape.

Interface (matches CFDBenchPatchDataset / PatchFlowFieldDataset):
    input  : (B, P, N*C)  – flattened patch features at current timestep
    output : (B, P, N*C)  – predicted flattened patch features at next timestep
    mask   : (B, P, N)    – optional valid-point mask (passed through, not used
                             internally but available for loss masking)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .condition_encoders import build_condition_encoder


# ============================================================
# 1.  Transformer building blocks
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.scale     = self.d_head ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, N, d_head)
        q, k, v = qkv.unbind(0)                    # each (B, H, N, d_head)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (ViT style)."""

    def __init__(self, d_model: int, n_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim    = int(d_model * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 2.  Patch Embedding  (replaces Conv2d patch embed in original)
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Linear projection of pre-split patches + learnable positional embedding.
    Supports optional parameter embedding projected from params_dim to d_model.

    Input  : (B, P, flattened_dim)   where flattened_dim = N_points × C_channels
    Output : (B, P+d, d_model) where d=1 if params_embed provided, else (B, P, d_model)
    """

    def __init__(self, flattened_dim: int, d_model: int, max_patches: int = 1024,
                 dropout: float = 0.0, params_dim: Optional[int] = None,
                 condition_encoder: str = 'token'):
        super().__init__()
        self.proj    = nn.Linear(flattened_dim, d_model)
        self.params_dim = params_dim
        self.condition_encoder_type = condition_encoder
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
        self.pos_emb = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.drop    = nn.Dropout(dropout)
        self.max_patches = max_patches
        self.d_model = d_model

    def forward(self, x: torch.Tensor, params_embed: torch.Tensor = None) -> torch.Tensor:
        B, P, _ = x.shape
        x = self.proj(x)                            # (B, P, d_model)

        if P <= self.max_patches:
            pos = self.pos_emb[:, :P, :]
        else:
            pos = F.interpolate(
                self.pos_emb.permute(0, 2, 1),
                size=P, mode='linear', align_corners=False,
            ).permute(0, 2, 1)

        x = x + pos
        x = self.drop(x)

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
                x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            else:
                condition_token = (
                    self.params_proj(params_embed)
                    if self.condition_encoder_type == 'token'
                    else self.condition_module(params_embed)
                )
                x = torch.cat([condition_token.unsqueeze(1), x], dim=1)

        return x


# ============================================================
# 3.  ViT Encoder with intermediate feature hooks
# ============================================================

class ViTEncoder(nn.Module):
    """
    Transformer encoder that exposes activations from 4 intermediate layers.

    The hook indices are chosen to create a 4-level feature pyramid at
    roughly 1/4, 1/2, 3/4, and 4/4 of total depth — matching DPT's
    default hooks=[2, 5, 8, 11] for a 12-layer ViT-B.

    Attributes:
        hooks (list[int]): block indices from which to extract features.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 hooks: Optional[List[int]] = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Default: 4 hooks at 1/4, 1/2, 3/4, end of network
        # Ensure n_layers >= 4 for valid hooks
        if hooks is None:
            if n_layers < 4:
                # When n_layers < 4, distribute hooks evenly
                hooks = [i * (n_layers - 1) // 3 for i in range(4)]
            else:
                q = n_layers // 4
                hooks = [q - 1, 2 * q - 1, 3 * q - 1, n_layers - 1]
        self.hooks = hooks

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns the 4 intermediate activations at self.hooks,
        as well as the final normalised output.

        Returns: (feat1, feat2, feat3, feat4)  each (B, P, d_model)
        """
        feats = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.hooks:
                feats[i] = x

        x = self.norm(x)
        # Replace last hook activation with the normalised output
        feats[self.hooks[-1]] = x

        return tuple(feats[h] for h in self.hooks)


# ============================================================
# 4.  Reassemble blocks  (replaces 2-D act_postprocess in original)
# ============================================================

class Reassemble(nn.Module):
    """
    Projects (B, P, d_model) → (B, out_channels, P') via a 1-D conv.

    Stride > 1 reduces the P dimension (analogous to downsampling in 2-D).
    Stride < 1 (fractional) is not supported; use stride=1 for the finest level.

    The original DPT uses strides {4, 2, 1, 0.5} mapped to 4 levels:
      level 1 (shallowest) : stride=4  (most downsampled)
      level 2              : stride=2
      level 3              : stride=1
      level 4 (deepest)    : stride=0.5 (upsampled via ConvTranspose)

    Here we mirror that with Conv1d strides {4, 2, 1} and a Conv1dTranspose
    with stride 2 for level 4, keeping the sequence structure intact.
    """

    def __init__(self, d_model: int, out_channels: int, stride: int = 1,
                 use_transpose: bool = False):
        super().__init__()
        self.use_transpose = use_transpose

        if use_transpose:
            # Upsample: ConvTranspose1d with stride=2 (level 4 in DPT)
            self.conv = nn.ConvTranspose1d(d_model, out_channels,
                                           kernel_size=2, stride=2)
        else:
            # Downsample or keep: Conv1d with given stride
            padding = 1 if stride == 1 else 0
            self.conv = nn.Conv1d(d_model, out_channels,
                                  kernel_size=stride if stride > 1 else 1,
                                  stride=max(stride, 1),
                                  padding=padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, P, d_model)
        returns : (B, P', out_channels)   – channel-last for fusion
        """
        x = x.permute(0, 2, 1)              # (B, d_model, P)
        x = self.conv(x)                     # (B, out_channels, P')
        x = x.permute(0, 2, 1)              # (B, P', out_channels)
        return self.norm(x)


# ============================================================
# 5.  1-D Residual Conv Unit  (replaces ResidualConvUnit_custom)
# ============================================================

class ResidualConvUnit1D(nn.Module):
    """
    Two 1-D conv layers with residual skip, replacing the 2-D version in DPT.

    Operates on (B, P, features).
    """

    def __init__(self, features: int, use_bn: bool = False):
        super().__init__()
        self.use_bn = use_bn
        self.conv1  = nn.Conv1d(features, features, kernel_size=3, padding=1, bias=not use_bn)
        self.conv2  = nn.Conv1d(features, features, kernel_size=3, padding=1, bias=not use_bn)
        if use_bn:
            self.bn1 = nn.BatchNorm1d(features)
            self.bn2 = nn.BatchNorm1d(features)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, P, features)  →  (B, P, features)"""
        res = x
        x = x.permute(0, 2, 1)            # (B, features, P)
        out = self.act(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = out.permute(0, 2, 1)         # (B, P, features)
        return out + res


# ============================================================
# 6.  Feature Fusion Block  (replaces FeatureFusionBlock_custom)
# ============================================================

class FeatureFusionBlock1D(nn.Module):
    """
    1-D feature fusion block.

    Fuses up to two feature maps (from adjacent levels), applies two
    residual conv units, then upsamples along the P dimension by 2×
    using linear interpolation (matching the bilinear upsampling in DPT).

    After upsampling, a 1×1 linear projection optionally halves channels
    when expand=True (not used by default).
    """

    def __init__(self, features: int, use_bn: bool = False, expand: bool = False):
        super().__init__()
        out_features = features // 2 if expand else features
        self.expand  = expand

        self.rcu1    = ResidualConvUnit1D(features, use_bn)
        self.rcu2    = ResidualConvUnit1D(features, use_bn)
        self.out_proj = nn.Linear(features, out_features)

    def forward(self, x: torch.Tensor,
                skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : (B, P, features)  – current (deeper) feature map
        skip : (B, P, features)  – optional feature from the shallower level
                                   (must be interpolated to same P before passing)
        Returns : (B, P*2, out_features)
        """
        out = x
        if skip is not None:
            out = out + self.rcu1(skip)

        out = self.rcu2(out)

        # Upsample along sequence dimension by 2×
        out = out.permute(0, 2, 1)         # (B, features, P)
        out = F.interpolate(out, scale_factor=2.0, mode='linear', align_corners=False)
        out = out.permute(0, 2, 1)         # (B, P*2, features)

        out = self.out_proj(out)            # (B, P*2, out_features)
        return out


# ============================================================
# 7.  Scratch (projection layers – replaces _make_scratch)
# ============================================================

class Scratch1D(nn.Module):
    """
    Channel projection layers that map from reassembled feature channels
    to a uniform `features` width – then the 4 refinenet fusion blocks.

    layer*_rn : (B, P', in_ch) → (B, P', features)  via Linear
    refinenet* : FeatureFusionBlock1D
    output_conv: (B, P_out, features) → (B, P_out, out_dim)  via Linear
    """

    def __init__(self, in_shapes: List[int], features: int,
                 out_dim: int, use_bn: bool = False):
        super().__init__()
        self.layer1_rn = nn.Linear(in_shapes[0], features)
        self.layer2_rn = nn.Linear(in_shapes[1], features)
        self.layer3_rn = nn.Linear(in_shapes[2], features)
        self.layer4_rn = nn.Linear(in_shapes[3], features)

        self.refinenet4 = FeatureFusionBlock1D(features, use_bn)
        self.refinenet3 = FeatureFusionBlock1D(features, use_bn)
        self.refinenet2 = FeatureFusionBlock1D(features, use_bn)
        self.refinenet1 = FeatureFusionBlock1D(features, use_bn)

        self.output_conv = nn.Sequential(
            nn.Linear(features, features // 2),
            nn.GELU(),
            nn.Linear(features // 2, out_dim),
        )


# ============================================================
# 8.  Full DPT model
# ============================================================

class DPT(nn.Module):
    """
    Dense Prediction Transformer adapted for pre-split patch sequences.

    Input  : (B, P, in_flattened_dim)   where in_flattened_dim = N_points × C_in
    Output : (B, P, out_flattened_dim)  predicted at next timestep

    Architecture
    ─────────────
    PatchEmbedding  →  ViTEncoder (n_layers Transformer blocks, 4 hooks)
    ↓
    4 × Reassemble  (project + optional stride change along P)
    ↓
    4 × Linear channel projection  (→ uniform `features` width)
    ↓
    Bottom-up Feature Fusion (refinenet4 → refinenet1, each ×2 upsampling)
    ↓
    Output head  (Linear → GELU → Linear → out_flattened_dim)

    The multi-scale fusion ensures that both local (shallow) and global
    (deep) Transformer features contribute to every output position,
    faithfully preserving the DPT design intent.

    Args:
        in_flattened_dim (int):  N_points × C_in per patch (model input dim).
        out_flattened_dim (int): N_points × C_out per patch (model output dim).
                                  If None, defaults to in_flattened_dim.
        features (int):      Uniform feature width in the fusion decoder. Default 256.
        d_model (int):       Transformer hidden dimension. Default 256.
        n_heads (int):       Number of attention heads. Default 8.
        n_layers (int):      Total Transformer depth. Default 12.
        mlp_ratio (float):   MLP expansion ratio inside each block. Default 4.0.
        dropout (float):     Dropout rate. Default 0.1.
        use_bn (bool):       Use BatchNorm in residual conv units. Default False.
        hooks (list[int]):   Block indices for intermediate feature extraction.
                             Defaults to [n//4-1, n//2-1, 3n//4-1, n-1].
        max_patches (int):   Maximum number of patches for positional embedding.
                              Interpolated if P > max_patches. Default 1024.
        params_dim (int):    Optional raw LLM embedding dimension. If set, the
                              parameter embedding is projected to d_model inside
                              the model and trained with the task loss.
    """

    def __init__(
        self,
        in_flattened_dim: int,
        out_flattened_dim: Optional[int] = None,
        features:      int   = 256,
        d_model:       int   = 256,
        n_heads:       int   = 8,
        n_layers:      int   = 12,
        mlp_ratio:     float = 4.0,
        dropout:       float = 0.1,
        use_bn:        bool  = False,
        hooks:         Optional[List[int]] = None,
        max_patches:   int   = 1024,
        params_dim:    Optional[int] = None,
        condition_encoder: str = 'token',
    ):
        super().__init__()
        if out_flattened_dim is None:
            out_flattened_dim = in_flattened_dim

        self.in_flattened_dim = in_flattened_dim
        self.out_flattened_dim = out_flattened_dim
        self.d_model       = d_model
        self.params_dim    = params_dim
        self.condition_encoder_type = condition_encoder

        # ── Encoder ──────────────────────────────────────────
        self.patch_embed = PatchEmbedding(in_flattened_dim, d_model,
                                          max_patches=max_patches,
                                          dropout=dropout,
                                          params_dim=params_dim,
                                          condition_encoder=condition_encoder)
        self.encoder = ViTEncoder(d_model, n_heads, n_layers,
                                  mlp_ratio=mlp_ratio, dropout=dropout,
                                  hooks=hooks)

        # ── Reassemble  (4 levels, coarser → finer) ──────────
        # Level 1 (shallowest hook): most downsampled  → stride=4
        # Level 2                  :                     stride=2
        # Level 3                  :                     stride=1
        # Level 4 (deepest hook)   : upsampled          → ConvTranspose1d ×2
        #
        # This recreates the spatial pyramid from the original DPT.
        re_ch = features   # all reassemble outputs share the same channel width
        self.reassemble1 = Reassemble(d_model, re_ch, stride=4,  use_transpose=False)
        self.reassemble2 = Reassemble(d_model, re_ch, stride=2,  use_transpose=False)
        self.reassemble3 = Reassemble(d_model, re_ch, stride=1,  use_transpose=False)
        self.reassemble4 = Reassemble(d_model, re_ch, stride=1,  use_transpose=True)

        # ── Scratch (projection + fusion) ────────────────────
        self.scratch = Scratch1D(
            in_shapes=[re_ch, re_ch, re_ch, re_ch],
            features=features,
            out_dim=out_flattened_dim,
            use_bn=use_bn,
        )

        self._init_weights()

    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    def _align_sequence(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Interpolate (B, P, C) along P to target_len."""
        if x.shape[1] == target_len:
            return x
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x.permute(0, 2, 1)

    # ----------------------------------------------------------
    def forward(
        self,
        x:    torch.Tensor,                         # (B, P, N*C)
        mask: Optional[torch.Tensor] = None,        # (B, P, N) – unused internally
        params_embed: Optional[torch.Tensor] = None, # (B, params_dim) – optional
    ) -> torch.Tensor:
        """
        Args:
            x    : (B, P, in_flattened_dim) – input patch features
            mask : (B, P, N) – optional validity mask (not used inside the model;
                               provided for interface consistency so callers can
                               apply it to the loss externally).
            params_embed : (B, params_dim) – optional pre-computed LLM embedding

        Returns:
            (B, P, out_flattened_dim) – predicted output patch features
        """
        B, P, _ = x.shape

        # ── 1. Embed patches ──────────────────────────────────
        tokens = self.patch_embed(x, params_embed)  # (B, P+d, d_model) if embed provided

        # ── 2. Encode and collect 4 feature maps ─────────────
        feat1, feat2, feat3, feat4 = self.encoder(tokens)
        # Each feat_i : (B, P+d, d_model) if embed provided

        # ── 3. Reassemble (multi-scale feature pyramid) ──────
        r1 = self.reassemble1(feat1)   # (B, P//4,  features)  shallowest
        r2 = self.reassemble2(feat2)   # (B, P//2,  features)
        r3 = self.reassemble3(feat3)   # (B, P,     features)
        r4 = self.reassemble4(feat4)   # (B, P*2,   features)  deepest

        # ── 4. Channel projection ─────────────────────────────
        l1 = self.scratch.layer1_rn(r1)  # (B, P//4, features)
        l2 = self.scratch.layer2_rn(r2)  # (B, P//2, features)
        l3 = self.scratch.layer3_rn(r3)  # (B, P,    features)
        l4 = self.scratch.layer4_rn(r4)  # (B, P*2,  features)

        # ── 5. Bottom-up fusion (refinenet4 → refinenet1) ────
        path4 = self.scratch.refinenet4(l4)

        path4_aligned = self._align_sequence(path4, l3.shape[1])
        path3 = self.scratch.refinenet3(l3, path4_aligned)

        path3_aligned = self._align_sequence(path3, l2.shape[1])
        path2 = self.scratch.refinenet2(l2, path3_aligned)

        path2_aligned = self._align_sequence(path2, l1.shape[1])
        path1 = self.scratch.refinenet1(l1, path2_aligned)

        # ── 6. Align output back to original P ───────────────
        out = self._align_sequence(path1, P)        # (B, P, features)

        # ── 7. Output projection ──────────────────────────────
        out = self.scratch.output_conv(out)         # (B, P, out_flattened_dim)

        return out


# ============================================================
# 9.  Loss wrapper
# ============================================================

class DPTLoss(nn.Module):
    """
    MSE loss for patch predictions, with optional mask support.

    Args:
        reduction (str): 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred:   torch.Tensor,                       # (B, P, N*C)
        target: torch.Tensor,                       # (B, P, N*C)
        mask:   Optional[torch.Tensor] = None,      # (B, P, N) bool
    ) -> torch.Tensor:
        """
        If mask is provided, expand it to match N*C features and compute
        masked MSE (only valid points contribute to the loss).
        """
        if mask is not None:
            B, P, N = mask.shape
            C = pred.shape[-1] // N
            # mask: (B, P, N) → (B, P, N, 1) → (B, P, N*C) after repeat
            mask_expand = mask.unsqueeze(-1).repeat(1, 1, 1, C)   # (B, P, N, C)
            mask_expand = mask_expand.reshape(B, P, N * C)         # (B, P, N*C)
            diff = (pred - target) ** 2
            loss = (diff * mask_expand.float()).sum() / (mask_expand.float().sum() + 1e-8)
        else:
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss

    @staticmethod
    def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.linalg.norm(pred.flatten() - target.flatten())
        nrm  = torch.linalg.norm(target.flatten()) + 1e-8
        return diff / nrm
