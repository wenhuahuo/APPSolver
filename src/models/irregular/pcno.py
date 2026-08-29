"""
Point Cloud Neural Operator (PCNO) for Irregular Mesh Flow Field Prediction (PyTorch)

Original implementation:
  https://github.com/PKU-CMEGroup/NeuralOperator/blob/main/pcno/pcno.py

Architecture  (u' = W·u + K·u + D·u  at each layer):
  W  – pointwise linear (Conv1d 1x1)
  K  – spectral integral operator with learnable Fourier modes and adaptive length scales
  D  – gradient-based differential operator (least-square gradient via scatter_add)

The three operators are summed at each layer and followed by a non-linearity,
mirroring the original "W + K + D" formulation exactly.

Data interface (matches CFDBenchIrregularDataset / IrregularFlowFieldDataset):
    pos : (B, N, 2)   – spatial coordinates
    fx  : (B, N, C)   – flow features at current timestep
    y   : (B, N, C)   – flow features at next timestep (prediction target)

The model receives `pos` and `fx` directly.  All geometric auxiliary data
(node weights, directed edges, gradient weights) are precomputed once per
dataset with `build_aux_from_pos()` and cached; they are stored as collated
batch tensors inside the DataLoader via `PCNOCollateFn`.

Precomputation pipeline (pure numpy, runs offline):
    aux = build_aux_from_pos(pos_np, k_neighbors=8, nmeasures=1)
    → node_mask          : (N, 1)           int
    → node_weights       : (N, nmeasures)   float32
    → directed_edges     : (E, 2)           int64
    → edge_gradient_wts  : (E, 2)           float32

Reference:
  "Point Cloud Neural Operator" – PKU-CMEGroup/NeuralOperator
"""

import hashlib
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Helpers – Fourier mode generation
# ---------------------------------------------------------------------------


def _compute_fourier_modes_2d(nks: list[int], Ls: list[float]) -> np.ndarray:
    """
    Build the canonical half-plane set of 2-D Fourier wavenumber pairs
    k = (2π/Lx · kx,  2π/Ly · ky),  sorted by |k|.

    Returns  k_pairs : float[nmodes, 2]
    """
    nx, ny = nks
    Lx, Ly = Ls
    pairs, mags = [], []
    for kx in range(-nx, nx + 1):
        for ky in range(ny + 1):
            if ky == 0 and kx <= 0:
                continue
            k = np.array([2 * math.pi / Lx * kx, 2 * math.pi / Ly * ky])
            pairs.append(k)
            mags.append(np.linalg.norm(k))
    pairs = np.array(pairs, dtype=np.float32)
    order = np.argsort(mags, kind="stable")
    return pairs[order]


def compute_fourier_modes(ndims: int, nks: list[int], Ls: list[float]) -> np.ndarray:
    """
    Compute `nmeasures` sets of Fourier modes.

    nks : int[ndims * nmeasures]
    Ls  : float[ndims * nmeasures]
    Returns  k_pairs : float[nmodes, ndims, nmeasures]
    """
    assert len(nks) == len(Ls)
    nmeasures = len(nks) // ndims
    stacks = []
    for i in range(nmeasures):
        k = _compute_fourier_modes_2d(
            nks[i * ndims : (i + 1) * ndims],
            Ls[i * ndims : (i + 1) * ndims],
        )
        stacks.append(k)
    # align to same nmodes (truncate to minimum)
    min_modes = min(k.shape[0] for k in stacks)
    stacks = [k[:min_modes] for k in stacks]
    return np.stack(stacks, axis=-1)  # (nmodes, ndims, nmeasures)


# ---------------------------------------------------------------------------
# Helpers – Fourier bases
# ---------------------------------------------------------------------------


def compute_fourier_bases(
    nodes: torch.Tensor,  # (B, N, ndims)
    modes: torch.Tensor,  # (nmodes, ndims, nmeasures)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute cos / sin / const Fourier bases.

    Returns
        bases_c, bases_s : (B, N, nmodes, nmeasures)
        bases_0          : (B, N, 1,      nmeasures)
    """
    temp = torch.einsum("bxd,kdw->bxkw", nodes, modes)  # (B, N, nmodes, nmeasures)
    bases_c = torch.cos(temp)
    bases_s = torch.sin(temp)
    B, N, _, nmeasures = temp.shape
    bases_0 = torch.ones(B, N, 1, nmeasures, dtype=temp.dtype, device=temp.device)
    return bases_c, bases_s, bases_0


# ---------------------------------------------------------------------------
# Scaled sigmoid / logit (for learnable length scales)
# ---------------------------------------------------------------------------


def scaled_sigmoid(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def scaled_logit(y: float, lo: float, hi: float) -> float:
    value = torch.tensor(float(y))
    return float(torch.log((value - lo) / (hi - value)))


# ---------------------------------------------------------------------------
# Spectral convolution layer  (K operator)
# ---------------------------------------------------------------------------


class SpectralConv(nn.Module):
    """
    Integral operator K via Fourier decomposition on point cloud.

    forward inputs / outputs match the original exactly:
        x        : (B, in_ch, N)
        bases_*  : (B, N, nmodes, nmeasures)
        → output : (B, out_ch, N)
    """

    def __init__(self, in_channels: int, out_channels: int, modes: torch.Tensor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        nmodes, ndims, nmeasures = modes.shape
        self.register_buffer("modes", modes)
        self.nmeasures = nmeasures
        scale = 1.0 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, nmodes, nmeasures)
        )
        self.weights_s = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, nmodes, nmeasures)
        )
        self.weights_0 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, 1, nmeasures)
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, in_ch, N)
        bases_c: torch.Tensor,  # (B, N, nmodes, nmeasures)
        bases_s: torch.Tensor,
        bases_0: torch.Tensor,
        wbases_c: torch.Tensor,
        wbases_s: torch.Tensor,
        wbases_0: torch.Tensor,
    ) -> torch.Tensor:
        # Forward transform (quadrature)
        x_c_hat = torch.einsum("bix,bxkw->bikw", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bxkw->bikw", x, wbases_s)
        x_0_hat = torch.einsum("bix,bxkw->bikw", x, wbases_0)

        wc, ws, w0 = self.weights_c, self.weights_s, self.weights_0

        # Multiply in frequency space (complex product: (a+ib)(c+id))
        f_c_hat = torch.einsum("bikw,iokw->bokw", x_c_hat, wc) - torch.einsum(
            "bikw,iokw->bokw", x_s_hat, ws
        )
        f_s_hat = torch.einsum("bikw,iokw->bokw", x_s_hat, wc) + torch.einsum(
            "bikw,iokw->bokw", x_c_hat, ws
        )
        f_0_hat = torch.einsum("bikw,iokw->bokw", x_0_hat, w0)

        # Inverse transform (synthesis)
        out = (
            torch.einsum("bokw,bxkw->box", f_0_hat, bases_0)
            + 2 * torch.einsum("bokw,bxkw->box", f_c_hat, bases_c)
            - 2 * torch.einsum("bokw,bxkw->box", f_s_hat, bases_s)
        )
        return out


# ---------------------------------------------------------------------------
# Gradient operator  (D operator)
# ---------------------------------------------------------------------------


def compute_gradient(
    f: torch.Tensor,  # (B, in_ch, N)
    directed_edges: torch.Tensor,  # (B, E, 2)  int
    edge_gradient_weights: torch.Tensor,  # (B, E, ndims)
) -> torch.Tensor:
    """
    Least-square gradient via message passing (scatter_add).

    Returns  f_gradients : (B, in_ch * ndims, N)
    """
    f = f.permute(0, 2, 1)  # (B, N, in_ch)
    B, N, C = f.shape
    _, E, ndims = edge_gradient_weights.shape

    target = directed_edges[..., 0]  # (B, E)
    source = directed_edges[..., 1]  # (B, E)

    # f at source and target nodes
    b_idx = torch.arange(B, device=f.device).unsqueeze(1)  # (B, 1)
    f_src = f[b_idx, source]  # (B, E, C)
    f_tgt = f[b_idx, target]  # (B, E, C)
    df = f_src - f_tgt  # (B, E, C)

    # message = edge_weight ⊗ df  →  (B, E, C * ndims)
    message = torch.einsum("bed,bec->becd", edge_gradient_weights, df)
    message = message.reshape(B, E, C * ndims)

    # scatter_add onto target nodes
    grads = torch.zeros(B, N, C * ndims, dtype=f.dtype, device=f.device)
    idx = target.unsqueeze(-1).expand_as(message)  # (B, E, C*ndims)
    grads.scatter_add_(1, idx, message)

    return grads.permute(0, 2, 1)  # (B, C*ndims, N)


# ---------------------------------------------------------------------------
# Main PCNO model
# ---------------------------------------------------------------------------


class PCNO(nn.Module):
    modes_base: torch.Tensor
    """
    Point Cloud Neural Operator (PCNO).

    Operator layer:   u' = (W + K + D)(u)
      W – Conv1d(1×1) pointwise linear
      K – spectral integral operator (Fourier on point cloud)
      D – gradient-based differential operator

    Args:
        in_channels   (int):  input  channels C  (e.g. 2 for u,v velocity)
        out_channels  (int):  output channels    (usually == in_channels)
        modes (np.ndarray):   Fourier modes array (nmodes, 2, nmeasures)
                              Use `compute_fourier_modes()` to build.
        layers (list[int]):   hidden channel widths per layer, e.g. [64,64,64,64]
        fc_dim (int):         projection MLP hidden dim (0 → linear projection)
        nmeasures (int):      number of integration measures (typically 1)
        inv_L_scale_min (float): lower bound of adaptive length-scale factor
        inv_L_scale_max (float): upper bound
        train_inv_L_scale (str|bool): 'independently' | 'together' | False
        act (str):            activation function name
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        modes: np.ndarray | None = None,
        layers: list[int] | None = None,
        fc_dim: int = 128,
        nmeasures: int = 1,
        inv_L_scale_min: float = 0.5,
        inv_L_scale_max: float = 2.0,
        train_inv_L_scale="independently",
        act: str = "gelu",
    ):
        super().__init__()
        self.ndims = 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc_dim = fc_dim
        self.nmeasures = nmeasures
        self.inv_L_scale_min = inv_L_scale_min
        self.inv_L_scale_max = inv_L_scale_max
        self.train_inv_L_scale = train_inv_L_scale

        layers = layers or [64, 64, 64, 64]
        self.layers = layers

        # Fourier modes buffer
        if modes is None:
            # sensible default: 12 modes per dim, domain [0,1]^2
            modes = compute_fourier_modes(
                2, [12, 12] * nmeasures, [1.0, 1.0] * nmeasures
            )
        modes_t = torch.from_numpy(modes.astype(np.float32))  # (nmodes, 2, nmeasures)
        self.register_buffer("modes_base", modes_t)

        # Learnable inverse length-scale (constrained via sigmoid)
        init_latent = scaled_logit(1.0, inv_L_scale_min, inv_L_scale_max)
        self.inv_L_scale_latent = nn.Parameter(
            torch.full((2, nmeasures), init_latent),
            requires_grad=bool(train_inv_L_scale),
        )

        # Input lifting
        in_dim = in_channels + 2  # concat coordinates to input features
        self.fc0 = nn.Linear(in_dim, layers[0])

        # W + K + D layers
        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_sz, out_sz, modes_t)
                for in_sz, out_sz in zip(layers, layers[1:], strict=True)
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_sz, out_sz, 1)
                for in_sz, out_sz in zip(layers, layers[1:], strict=True)
            ]
        )
        self.gws = nn.ModuleList(
            [
                nn.Conv1d(2 * in_sz, out_sz, 1)  # gradient doubles channels
                for in_sz, out_sz in zip(layers, layers[1:], strict=True)
            ]
        )

        # Output projection
        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_channels)
        else:
            self.fc2 = nn.Linear(layers[-1], out_channels)

        self.act = getattr(F, act)
        self.softsign = F.softsign

        # Parameter groups for (possibly separate) optimisers
        self.normal_params = []
        self.inv_L_params = []
        for name, param in self.named_parameters():
            if param is self.inv_L_scale_latent:
                if train_inv_L_scale == "together":
                    self.normal_params.append(param)
                elif train_inv_L_scale == "independently":
                    self.inv_L_params.append(param)
                # else False → not added to any group (frozen)
            else:
                self.normal_params.append(param)

    # ------------------------------------------------------------------
    def _effective_modes(self) -> torch.Tensor:
        """Apply adaptive length-scale to stored Fourier modes."""
        inv_L = scaled_sigmoid(
            self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max
        )  # (2, nmeasures)
        return self.modes_base * inv_L  # broadcast (nmodes,2,nm)

    # ------------------------------------------------------------------
    def forward(
        self,
        pos: torch.Tensor,  # (B, N, 2)
        fx: torch.Tensor,  # (B, N, C)
        node_weights: torch.Tensor,  # (B, N, nmeasures)
        directed_edges: torch.Tensor,  # (B, E, 2)         int
        edge_gradient_weights: torch.Tensor,  # (B, E, 2)
        node_mask: torch.Tensor | None = None,  # (B, N, 1) or None
    ) -> torch.Tensor:
        """
        Returns predicted flow field: (B, N, out_channels)
        """
        # 1. Compute Fourier bases with current length-scale
        modes = self._effective_modes()  # (nmodes, 2, nmeasures)
        bases_c, bases_s, bases_0 = compute_fourier_bases(pos, modes)

        # Weight bases by node measure
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        # 2. Lift input  (concatenate coordinates as extra features)
        x = torch.cat([fx, pos], dim=-1)  # (B, N, C+2)
        x = self.fc0(x)  # (B, N, layers[0])
        x = x.permute(0, 2, 1)  # (B, layers[0], N)

        # 3. Operator layers
        n_layers = len(self.ws)
        for i, (sp, w, gw) in enumerate(
            zip(self.sp_convs, self.ws, self.gws, strict=True)
        ):
            x1 = sp(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw(
                self.softsign(
                    compute_gradient(x, directed_edges, edge_gradient_weights)
                )
            )
            x = x1 + x2 + x3
            if i < n_layers - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)  # (B, N, layers[-1])

        # 4. Output projection
        if self.fc_dim > 0:
            x = self.act(self.fc1(x))
        x = self.fc2(x)  # (B, N, out_channels)

        # 5. Apply node mask (zero out padding if any)
        if node_mask is not None:
            x = x * node_mask

        return x


# ---------------------------------------------------------------------------
# Loss wrapper
# ---------------------------------------------------------------------------


class PCNOLoss(nn.Module):
    """Masked MSE loss; use :meth:`relative_l2` for the relative-L2 metric."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)

    @staticmethod
    def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.linalg.norm(pred.flatten() - target.flatten())
        nrm = torch.linalg.norm(target.flatten()) + 1e-8
        return diff / nrm


# ---------------------------------------------------------------------------
# Geometric auxiliary data builder  (pure numpy, offline preprocessing)
# ---------------------------------------------------------------------------


def build_aux_from_pos(
    pos: np.ndarray,  # (N, 2)  coordinates of one sample
    k_neighbors: int = 8,
    nmeasures: int = 1,
) -> dict:
    """
    Build PCNO auxiliary data from a plain point cloud (no mesh connectivity).

    Strategy:
      - node_weights: Voronoi-cell area proxy via inverse-distance Shepard sum,
        then normalised to sum to 1.
      - directed_edges: k-nearest-neighbour graph (symmetric directed edges).
      - edge_gradient_weights: least-square gradient weights (pseudo-inverse of
        displacement matrix for each node).

    Args:
        pos         : (N, 2) float32 coordinate array for one sample.
        k_neighbors : number of nearest neighbours for the kNN graph.
        nmeasures   : number of integration measures (usually 1).

    Returns a dict with keys:
        node_mask              : (N, 1)            int32
        node_weights           : (N, nmeasures)    float32
        directed_edges         : (E, 2)            int64
        edge_gradient_weights  : (E, 2)            float32
    """
    N = pos.shape[0]

    # k+1 because query includes the node itself (distance 0)
    k = min(k_neighbors + 1, N)
    # pi-lens-ignore: python-sql-injection (scipy KD-tree nearest-neighbor query)
    dists, indices = KDTree(pos).query(pos, k=k)  # (N, k)
    indices = np.asarray(indices)

    # ------------------------------------------------------------------
    # Node weights: Shepard-style area estimate, normalised
    # ------------------------------------------------------------------
    # Use average k-NN distance squared as proxy for Voronoi cell area
    mean_dist_sq = np.mean(dists[:, 1:] ** 2, axis=1)  # exclude self (dist=0)
    mean_dist_sq = np.maximum(mean_dist_sq, 1e-12)
    raw_weight = mean_dist_sq  # (N,)
    node_weights = raw_weight / raw_weight.sum()  # normalise to sum=1
    node_weights = np.tile(node_weights[:, None], (1, nmeasures)).astype(np.float32)

    # ------------------------------------------------------------------
    # Directed edges: kNN (exclude self), preserving the original row order.
    # ------------------------------------------------------------------
    neighbors = np.asarray(indices[:, 1:], dtype=np.int64)
    neighbor_count = neighbors.shape[1]
    sources = np.repeat(np.arange(N, dtype=np.int64), neighbor_count)
    directed_edges = np.column_stack((sources, neighbors.reshape(-1)))

    # ------------------------------------------------------------------
    # Edge gradient weights: batched version of the original per-node SVD.
    # Chunking bounds temporary memory for large CFDBench cases.
    # ------------------------------------------------------------------
    edge_gradient_weights = np.empty(
        (N * neighbor_count, pos.shape[1]), dtype=np.float32
    )
    chunk_size = 20000
    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        chunk_neighbors = neighbors[start:stop]
        dx = pos[chunk_neighbors] - pos[start:stop, None, :]
        U, S, Vt = np.linalg.svd(dx, full_matrices=False)
        rcond = np.where(S[:, 0] > 0, 1e-3 * S[:, 0], 1e-12)
        S_inv = np.zeros_like(S)
        np.divide(1.0, S, out=S_inv, where=rcond[:, None] < S)
        pinvdx = (Vt.transpose(0, 2, 1) * S_inv[:, None, :]) @ U.transpose(0, 2, 1)
        edge_gradient_weights[start * neighbor_count : stop * neighbor_count] = (
            pinvdx.transpose(0, 2, 1).reshape(-1, pos.shape[1])
        )

    return {
        "node_mask": np.ones((N, 1), dtype=np.int32),
        "node_weights": node_weights,
        "directed_edges": directed_edges,
        "edge_gradient_weights": edge_gradient_weights,
    }


def _pcno_reference_hash(pos: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(pos, dtype=np.float32)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def save_pcno_aux_cache(
    cache_path: str,
    aux: dict,
    pos: np.ndarray,
    k_neighbors: int,
    nmeasures: int,
) -> None:
    """Atomically save geometry-bound PCNO auxiliary arrays."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temporary_path = f"{cache_path}.tmp-{os.getpid()}"
    with open(temporary_path, "wb") as handle:
        np.savez(
            handle,
            **aux,
            reference_hash=np.asarray(_pcno_reference_hash(pos)),
            k_neighbors=np.asarray(k_neighbors, dtype=np.int64),
            nmeasures=np.asarray(nmeasures, dtype=np.int64),
        )
    os.replace(temporary_path, cache_path)


def load_pcno_aux_cache(
    cache_path: str,
    pos: np.ndarray,
    k_neighbors: int,
    nmeasures: int,
) -> dict:
    """Load a PCNO auxiliary cache and verify its geometry and configuration."""
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Missing PCNO auxiliary cache: {cache_path}")

    required = {
        "node_mask",
        "node_weights",
        "directed_edges",
        "edge_gradient_weights",
        "reference_hash",
        "k_neighbors",
        "nmeasures",
    }
    with np.load(cache_path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"PCNO auxiliary cache missing {sorted(missing)}: {cache_path}"
            )
        cached_k = int(data["k_neighbors"])
        cached_nmeasures = int(data["nmeasures"])
        cached_hash = str(data["reference_hash"])
        aux = {
            key: data[key]
            for key in (
                "node_mask",
                "node_weights",
                "directed_edges",
                "edge_gradient_weights",
            )
        }

    if cached_k != k_neighbors or cached_nmeasures != nmeasures:
        raise ValueError(
            f"PCNO auxiliary cache configuration mismatch: {cache_path} "
            f"has k={cached_k}, nmeasures={cached_nmeasures}; "
            f"requested k={k_neighbors}, nmeasures={nmeasures}"
        )
    if cached_hash != _pcno_reference_hash(pos):
        raise ValueError(f"PCNO auxiliary cache geometry mismatch: {cache_path}")

    point_count = len(pos)
    edge_count = point_count * min(k_neighbors, max(0, point_count - 1))
    expected_shapes = {
        "node_mask": (point_count, 1),
        "node_weights": (point_count, nmeasures),
        "directed_edges": (edge_count, 2),
        "edge_gradient_weights": (edge_count, pos.shape[1]),
    }
    for key, shape in expected_shapes.items():
        if aux[key].shape != shape:
            raise ValueError(
                f"PCNO auxiliary cache shape mismatch for {key}: "
                f"expected {shape}, found {aux[key].shape} in {cache_path}"
            )
    return aux


def collate_aux_batch(aux_list: list[dict]) -> dict:
    """
    Pad and stack a list of per-sample aux dicts into batch tensors.

    All samples must have the same N (as guaranteed by CFDBenchIrregularDataset).
    Edges are padded to the maximum edge count across the batch.

    Returns dict of torch.Tensor:
        node_mask              : (B, N, 1)
        node_weights           : (B, N, nmeasures)
        directed_edges         : (B, E_max, 2)   int64
        edge_gradient_weights  : (B, E_max, 2)
    """
    B = len(aux_list)
    max_edges = max(a["directed_edges"].shape[0] for a in aux_list)
    ndims = aux_list[0]["edge_gradient_weights"].shape[1]
    nmeasures = aux_list[0]["node_weights"].shape[1]
    N = aux_list[0]["node_mask"].shape[0]

    node_mask = np.zeros((B, N, 1), dtype=np.int32)
    node_wts = np.zeros((B, N, nmeasures), dtype=np.float32)
    edges = np.zeros((B, max_edges, 2), dtype=np.int64)
    edge_wts = np.zeros((B, max_edges, ndims), dtype=np.float32)

    for i, a in enumerate(aux_list):
        E = a["directed_edges"].shape[0]
        node_mask[i] = a["node_mask"]
        node_wts[i] = a["node_weights"]
        edges[i, :E] = a["directed_edges"]
        edge_wts[i, :E] = a["edge_gradient_weights"]

    return {
        "node_mask": torch.from_numpy(node_mask).float(),
        "node_weights": torch.from_numpy(node_wts),
        "directed_edges": torch.from_numpy(edges),
        "edge_gradient_weights": torch.from_numpy(edge_wts),
    }
