"""
Metrics calculation for flow field prediction

Unified interface for both irregular (points) and patch (flattened patches) formats:
- Irregular: pred/target are [B, N, C], pass no mask (all points are valid)
- Patch: convert to [B, N, C] via patches_to_points first, then pass the valid_mask
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def patches_to_points(
    patches: torch.Tensor,
    quadtree,
    batch_size: int,
    n_points: int,
    n_channels: int,
    input_dim: int,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert flattened patches [B, P, N*C] back to points format [B, N, C].

    The patch tensor may contain concatenated coordinates and flow values
    (input_dim = coord_dim + flow_dim).  Only the first n_channels (flow)
    channels are returned so the result is directly comparable with the
    irregular format output.

    Args:
        patches:    [B, P, max_points * input_dim] flattened patch tensor
        quadtree:   QuadTreeMesh with patch point-index information
        batch_size: B
        n_points:   total number of mesh points N
        n_channels: number of flow output channels C (output_dim)
        input_dim:  channels per point stored in the patch (coord_dim + C)
        max_points: maximum number of points in any single patch

    Returns:
        points_values: [B, N, C] flow values at every mesh point
        valid_mask:    [B, N]  True for points that belong to at least one patch
    """
    device = patches.device

    points_values = torch.zeros(batch_size, n_points, n_channels, device=device)
    valid_mask    = torch.zeros(batch_size, n_points, dtype=torch.bool, device=device)

    for patch_idx, patch in enumerate(quadtree.patches):
        point_indices = patch.points
        n_pts = len(point_indices)

        # Extract the valid (non-padded) portion of this patch
        patch_flat     = patches[:, patch_idx, :n_pts * input_dim]          # [B, n_pts*input_dim]
        patch_reshaped = patch_flat.reshape(batch_size, n_pts, input_dim)   # [B, n_pts, input_dim]

        # Keep only the flow channels (first n_channels columns)
        patch_flows = patch_reshaped[:, :, :n_channels]                     # [B, n_pts, C]

        # Scatter into the full point array using advanced indexing (no Python loop over B)
        points_values[:, point_indices, :] = patch_flows
        valid_mask[:, point_indices]        = True

    return points_values, valid_mask


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute MAE, MSE and RMSE between pred and target.

    Both irregular and patch paths produce [B, N, C] tensors before calling
    this function, so the calculation is identical for both.

    Args:
        pred:   [B, N, C] predicted flow values (normalised space)
        target: [B, N, C] ground-truth flow values (normalised space)
        mask:   [B, N] boolean mask; True = valid point.
                Pass None for irregular data (every point is valid).
                Pass the mask returned by patches_to_points for patch data.

    Returns:
        dict with scalar float values for 'mae', 'mse', 'rmse'
    """
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)   # [B, N, C]
        n_valid = mask_expanded.sum().clamp(min=1)
        mse = ((pred - target) ** 2 * mask_expanded).sum() / n_valid
        mae = (torch.abs(pred - target) * mask_expanded).sum() / n_valid
    else:
        mse = ((pred - target) ** 2).mean()
        mae = torch.abs(pred - target).mean()

    return {
        'mae':  mae.item(),
        'mse':  mse.item(),
        'rmse': torch.sqrt(mse).item(),
    }


class MetricsCalculator:
    """
    Accumulates per-batch metrics and returns their mean over all batches.

    Usage (irregular):
        calc = MetricsCalculator()
        calc.update(pred, target)           # mask=None  →  all points counted

    Usage (patch):
        calc = MetricsCalculator()
        calc.update(pred_points, target_points, mask=combined_mask)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_mae = 0.0
        self.total_mse = 0.0
        self.n_batches  = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor,
               mask: Optional[torch.Tensor] = None):
        """
        Args:
            pred:   [B, N, C]
            target: [B, N, C]
            mask:   [B, N] or None
        """
        m = compute_metrics(pred, target, mask)
        self.total_mae += m['mae']
        self.total_mse += m['mse']
        self.n_batches  += 1

    def compute(self) -> Dict[str, float]:
        if self.n_batches == 0:
            return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
        avg_mae = self.total_mae / self.n_batches
        avg_mse = self.total_mse / self.n_batches
        return {
            'mae':  avg_mae,
            'mse':  avg_mse,
            'rmse': float(np.sqrt(avg_mse)),
        }


if __name__ == '__main__':
    pred   = torch.randn(2, 1000, 4)
    target = torch.randn(2, 1000, 4)
    mask   = torch.randint(0, 2, (2, 1000)).bool()

    m = compute_metrics(pred, target, mask)
    print(f"With mask:    MAE={m['mae']:.4f}, MSE={m['mse']:.4f}, RMSE={m['rmse']:.4f}")

    m2 = compute_metrics(pred, target, None)
    print(f"Without mask: MAE={m2['mae']:.4f}, MSE={m2['mse']:.4f}, RMSE={m2['rmse']:.4f}")