"""
Fusion-DeepONet for Irregular Mesh Flow Field Prediction (PyTorch)

Original JAX implementation:
  https://github.com/ahmadpeyvan/Fusion-DeepONet/blob/main/src/Semi_ellipse/unstructured_grid/fusion_deeponet.py

Architecture:
  - Branch Net: encodes the input function (current flow field fx at all nodes)
    using a "fused" MLP with skip connections and mixed tanh+sin activations.
  - Trunk Net:  encodes query coordinates (pos) with the same fused MLP style.
  - Output:     einsum dot-product between trunk and branch embeddings.

Data interface (matches CFDBenchIrregularDataset / ShipIrregularDataset):
    pos : (B, N, 2)   – spatial coordinates
    fx  : (B, N, C)   – flow features at current timestep  [branch input]
    y   : (B, N, C)   – flow features at next timestep     [prediction target]

The branch net receives the *flattened* flow field (B, N*C) and produces
(B, C * G_dim) embeddings. The trunk net receives each query point (B, N, 2)
and produces (B, N, G_dim) embeddings. The final prediction at each point is
the inner product between the per-channel trunk slice and the branch vector,
matching the original einsum formulation.

Note: because N can vary across datasets, the branch net projects to a fixed
intermediate size via a learnable linear layer applied per-node before pooling,
so the model is resolution-invariant and works with any N.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fused activation layer: tanh + sin mixture (mirrors JAX implementation)
# ---------------------------------------------------------------------------

class FusedActivationLayer(nn.Module):
    """
    Single linear layer with mixed tanh/sin activation and learnable scale params.

    Implements:
        h = 10*a * tanh(10*A*Wx + c) + 10*a1 * sin(10*F1*Wx + c1)

    where A, c, a1, F1, c1 are scalar learnable parameters (one per layer).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        # Scalar learnable parameters (initialised to match JAX defaults)
        self.A  = nn.Parameter(torch.full((1,), 0.1))   # tanh amplitude scale
        self.c  = nn.Parameter(torch.full((1,), 0.1))   # tanh phase
        self.a1 = nn.Parameter(torch.full((1,), 0.0))   # sin amplitude
        self.F1 = nn.Parameter(torch.full((1,), 0.1))   # sin frequency scale
        self.c1 = nn.Parameter(torch.full((1,), 0.0))   # sin phase

        # Weight initialisation: Glorot normal
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        tanh_part = 10.0 * self.A  * torch.tanh(10.0 * self.A  * z + self.c)
        sin_part  = 10.0 * self.a1 * torch.sin(10.0 * self.F1 * z + self.c1)
        return tanh_part + sin_part


# ---------------------------------------------------------------------------
# Fused MLP with skip connections (mirrors fnn_fuse_mixed_add in JAX)
# ---------------------------------------------------------------------------

class FusedMLP(nn.Module):
    """
    MLP with fused tanh+sin activations and residual skip connections between
    hidden layers (skip[i] += skip[i-1]).

    layers: list of widths, e.g. [in, 64, 64, 64, out]
    The last layer is a plain linear (no activation).
    """

    def __init__(self, layers: list):
        super().__init__()
        assert len(layers) >= 2

        # Hidden fused layers (all but the last transition)
        n_hidden = len(layers) - 2
        self.fused_layers = nn.ModuleList([
            FusedActivationLayer(layers[i], layers[i + 1])
            for i in range(n_hidden)
        ])
        # Final linear output layer
        self.out_layer = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_normal_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        returns: (..., out_features)
        """
        skips = []
        h = x
        for i, layer in enumerate(self.fused_layers):
            h = layer(h)
            if i > 0:
                h = h + skips[-1]   # additive skip (matches JAX skip[i] += skip[i-1])
            skips.append(h)
        return self.out_layer(h)


# ---------------------------------------------------------------------------
# Fused Branch+Trunk with cross-fusion  (mirrors fnn_fuse_mixed_add)
# ---------------------------------------------------------------------------

class FusedBranchTrunk(nn.Module):
    """
    Mirrors the JAX function fnn_fuse_mixed_add:

        branch (Xb): accumulates hidden representations with skip sums.
        trunk  (Xt): at each hidden layer, element-wise multiplied with the
                     corresponding branch skip: inputst *= skip[i][:, None, :]

    This cross-fusion allows the trunk to be modulated by global branch info
    at every depth, which is the key novelty of Fusion-DeepONet.
    """

    def __init__(self, branch_layers: list, trunk_layers: list):
        super().__init__()
        assert len(branch_layers) == len(trunk_layers), \
            "Branch and trunk must have the same depth"

        n_hidden = len(branch_layers) - 2

        self.branch_hidden = nn.ModuleList([
            FusedActivationLayer(branch_layers[i], branch_layers[i + 1])
            for i in range(n_hidden)
        ])
        self.branch_out = nn.Linear(branch_layers[-2], branch_layers[-1])

        self.trunk_hidden = nn.ModuleList([
            FusedActivationLayer(trunk_layers[i], trunk_layers[i + 1])
            for i in range(n_hidden)
        ])
        self.trunk_out = nn.Linear(trunk_layers[-2], trunk_layers[-1])

        for linear in [self.branch_out, self.trunk_out]:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        xb: torch.Tensor,   # branch input: (B, branch_in)
        xt: torch.Tensor,   # trunk  input: (B, N, trunk_in)
    ):
        """
        Returns:
            yt: (B, N, trunk_out)   – trunk output (per query point)
            yb: (B, branch_out)     – branch output (global)
        """
        # --- Branch forward with skip accumulation ---
        branch_skips = []
        hb = xb
        for i, layer in enumerate(self.branch_hidden):
            hb = layer(hb)
            if i > 0:
                hb = hb + branch_skips[-1]
            branch_skips.append(hb)

        yb = self.branch_out(hb)

        # --- Trunk forward with cross-fusion from branch skips ---
        ht = xt
        for i, layer in enumerate(self.trunk_hidden):
            ht = layer(ht)
            # branch_skips[i]: (B, hidden) → (B, 1, hidden) for broadcasting
            ht = ht * branch_skips[i].unsqueeze(1)

        yt = self.trunk_out(ht)
        return yt, yb


# ---------------------------------------------------------------------------
# Branch encoder: node-wise projection + global pooling
# ---------------------------------------------------------------------------

class BranchEncoder(nn.Module):
    """
    Encodes the input flow field fx: (B, N, C_in) into a global vector (B, branch_dim).

    Strategy: apply a small per-node MLP then mean-pool over N.
    This makes the branch net resolution-invariant.
    """

    def __init__(self, in_channels: int, hidden_dim: int, branch_dim: int):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, branch_dim),
        )

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        """
        fx: (B, N, C_in)
        returns: (B, branch_dim)  – mean-pooled global representation
        """
        h = self.node_proj(fx)      # (B, N, branch_dim)
        return h.mean(dim=1)        # (B, branch_dim)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FusionDeepONet(nn.Module):
    """
    Fusion-DeepONet (PyTorch) for irregular mesh data.

    Args:
        in_channels  (int): number of input flow channels C  (e.g. 2 for u, v)
        out_channels (int): number of output channels        (same as in_channels for next-step prediction)
        hidden_dim   (int): width of all hidden layers       (default 64)
        n_layers     (int): number of hidden layers in branch/trunk  (default 3)
        G_dim        (int): trunk/branch output embedding dim per channel (default 64)
        coord_dim    (int): spatial coordinate dimension     (default 2)
    """

    def __init__(
        self,
        in_channels:  int = 2,
        out_channels: int = 2,
        hidden_dim:   int = 64,
        n_layers:     int = 3,
        G_dim:        int = 64,
        coord_dim:    int = 2,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.G_dim        = G_dim

        # Branch encoder: fx (B, N, C) → (B, branch_dim)
        branch_dim = hidden_dim
        self.branch_encoder = BranchEncoder(in_channels, hidden_dim, branch_dim)

        # Branch and trunk network layers
        # branch: branch_dim → [hidden]*n_layers → out_channels * G_dim
        # trunk:  coord_dim  → [hidden]*n_layers → G_dim
        branch_layers = [branch_dim] + [hidden_dim] * n_layers + [out_channels * G_dim]
        trunk_layers  = [coord_dim]  + [hidden_dim] * n_layers + [G_dim]

        self.fused_net = FusedBranchTrunk(branch_layers, trunk_layers)

    def forward(
        self,
        pos: torch.Tensor,   # (B, N, 2)  spatial coordinates
        fx:  torch.Tensor,   # (B, N, C)  input flow field
    ) -> torch.Tensor:
        """
        Returns predicted flow field at next timestep: (B, N, out_channels)
        """
        B, N, _ = pos.shape
        C_out    = self.out_channels
        G        = self.G_dim

        # 1. Encode input function → global branch vector
        branch_in = self.branch_encoder(fx)              # (B, branch_dim)

        # 2. Fused trunk + branch forward
        yt, yb = self.fused_net(branch_in, pos)          # yt: (B, N, G), yb: (B, C_out*G)

        # 3. Reshape branch output to (B, C_out, G)
        yb = yb.view(B, C_out, G)                        # (B, C_out, G)

        # 4. Dot product for each output channel:
        #    pred[b, n, c] = sum_g  yb[b, c, g] * yt[b, n, g]
        #    equivalent to the original einsum: 'ijkl,inkm->inl'
        pred = torch.einsum('bcg,bng->bnc', yb, yt)      # (B, N, C_out)

        return pred


# ---------------------------------------------------------------------------
# Loss wrapper (consistent with other models in this directory)
# ---------------------------------------------------------------------------

class FusionDeepONetLoss(nn.Module):
    """
    MSE loss with optional relative L2 metric logging.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred:   torch.Tensor,   # (B, N, C)
        target: torch.Tensor,   # (B, N, C)
    ) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)

    @staticmethod
    def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Relative L2 error (scalar)."""
        diff_norm   = torch.linalg.norm(pred.flatten() - target.flatten())
        target_norm = torch.linalg.norm(target.flatten())
        return diff_norm / (target_norm + 1e-8)
