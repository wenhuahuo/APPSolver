"""Point recovery and metrics for flow-field predictions."""

import math

import torch

MetricValue = float | list[float]


def patches_to_points(
    patches: torch.Tensor,
    quadtree,
    batch_size: int,
    n_points: int,
    n_channels: int,
    input_dim: int,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert flattened patches [B, P, N*C] back to points format [B, N, C].

    The patch tensor may contain concatenated coordinates and flow values
    (input_dim = coord_dim + flow_dim). Only the first ``n_channels`` values
    for each point are returned. ``max_points`` remains part of the public
    signature for compatibility with existing callers.
    """
    del max_points  # Padding is already excluded using each patch's point count.
    device = patches.device
    points_values = torch.zeros(
        batch_size, n_points, n_channels, device=device, dtype=patches.dtype
    )
    valid_mask = torch.zeros(batch_size, n_points, dtype=torch.bool, device=device)

    for patch_idx, patch in enumerate(quadtree.patches):
        point_indices = patch.points
        n_pts = len(point_indices)
        patch_flat = patches[:, patch_idx, : n_pts * input_dim]
        patch_reshaped = patch_flat.reshape(batch_size, n_pts, input_dim)
        patch_flows = patch_reshaped[:, :, :n_channels]
        points_values[:, point_indices, :] = patch_flows
        valid_mask[:, point_indices] = True

    return points_values, valid_mask


def recover_points_knn(
    point_predictions: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_weights: torch.Tensor,
    sampled_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover predictions at all points using a precomputed global k-NN map.

    Args:
        point_predictions: ``[S, C]`` or ``[B, S, C]`` sampled predictions. If
            ``sampled_indices`` is omitted, this must instead be a scattered
            full-size tensor ``[N, C]`` or ``[B, N, C]``.
        neighbor_indices: ``[N, K]`` or ``[B, N, K]`` neighbor indices in the
            *full* point indexing (not positions in the compact sampled array).
        neighbor_weights: weights with the same shape as ``neighbor_indices``.
            They are normalized per output point.
        sampled_indices: optional ``[S]`` or ``[B, S]`` full indices of compact
            predictions. Recovered values at these points are overwritten with
            the corresponding predictions, preserving them exactly.

    Returns:
        Recovered predictions shaped ``[N, C]`` or ``[B, N, C]``. The operation
        uses only PyTorch gather/scatter operations and is differentiable with
        respect to predictions and weights.
    """
    if point_predictions.ndim not in (2, 3):
        raise ValueError("point_predictions must have shape [S, C] or [B, S, C]")
    if neighbor_indices.ndim not in (2, 3):
        raise ValueError("neighbor_indices must have shape [N, K] or [B, N, K]")
    if neighbor_weights.shape != neighbor_indices.shape:
        raise ValueError("neighbor_weights must have the same shape as neighbor_indices")

    unbatched = point_predictions.ndim == 2
    predictions = point_predictions.unsqueeze(0) if unbatched else point_predictions
    indices = neighbor_indices.unsqueeze(0) if neighbor_indices.ndim == 2 else neighbor_indices
    weights = neighbor_weights.unsqueeze(0) if neighbor_weights.ndim == 2 else neighbor_weights

    batch_size, n_pred, n_channels = predictions.shape
    n_points = indices.shape[-2]
    if indices.shape[0] == 1 and batch_size != 1:
        indices = indices.expand(batch_size, -1, -1)
        weights = weights.expand(batch_size, -1, -1)
    elif indices.shape[0] != batch_size:
        raise ValueError("neighbor map batch dimension does not match predictions")

    indices = indices.to(device=predictions.device, dtype=torch.long)
    weights = weights.to(device=predictions.device, dtype=predictions.dtype)
    if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= n_points):
        raise ValueError("neighbor_indices contains an out-of-range full point index")

    compact_predictions = None
    sample_idx = None
    if sampled_indices is not None:
        sample_idx = sampled_indices.unsqueeze(0) if sampled_indices.ndim == 1 else sampled_indices
        if sample_idx.ndim != 2:
            raise ValueError("sampled_indices must have shape [S] or [B, S]")
        if sample_idx.shape[0] == 1 and batch_size != 1:
            sample_idx = sample_idx.expand(batch_size, -1)
        if sample_idx.shape[0] != batch_size:
            raise ValueError("sampled_indices batch dimension does not match predictions")
        sample_idx = sample_idx.to(device=predictions.device, dtype=torch.long)
        if sample_idx.numel() and (sample_idx.min().item() < 0 or sample_idx.max().item() >= n_points):
            raise ValueError("sampled_indices contains an out-of-range point index")

        if n_pred == sample_idx.shape[1]:
            compact_predictions = predictions
            source = predictions.new_zeros(batch_size, n_points, n_channels).scatter(
                1, sample_idx.unsqueeze(-1).expand(-1, -1, n_channels), predictions
            )
        elif n_pred == n_points:
            source = predictions
        else:
            raise ValueError("compact predictions and sampled_indices have different lengths")
    elif n_pred == n_points:
        source = predictions
    else:
        raise ValueError("sampled_indices is required for compact sampled predictions")

    batch = torch.arange(batch_size, device=predictions.device)[:, None, None]
    neighbors = source[batch, indices]  # [B, N, K, C]
    denominator = weights.sum(dim=-1, keepdim=True)
    safe_denominator = denominator.clamp_min(torch.finfo(weights.dtype).eps)
    normalized_weights = weights / safe_denominator
    recovered = (neighbors * normalized_weights.unsqueeze(-1)).sum(dim=-2)

    if sample_idx is not None:
        exact = compact_predictions
        if exact is None:
            exact = source.gather(1, sample_idx.unsqueeze(-1).expand(-1, -1, n_channels))
        recovered = recovered.scatter(
            1, sample_idx.unsqueeze(-1).expand(-1, -1, n_channels), exact
        )

    return recovered.squeeze(0) if unbatched else recovered


class MetricsCalculator:
    """Accumulate pointwise errors over elements, rather than over batches."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total_abs_error = 0.0
        self.total_squared_error = 0.0
        self.total_target_squared = 0.0
        self.element_count = 0
        self.channel_abs_error: list[float] | None = None
        self.channel_squared_error: list[float] | None = None
        self.channel_target_squared: list[float] | None = None
        self.point_count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Add ``[B, N, C]`` predictions, optionally at masked valid points."""
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shapes")
        if pred.ndim == 0:
            raise ValueError("pred and target must contain at least one element")

        # A flattened tensor was accepted by the previous implementation and is
        # still used by one patch-evaluation path; treat it as a single channel.
        n_channels = pred.shape[-1] if pred.ndim > 1 else 1
        channel_abs_state = self.channel_abs_error
        channel_sq_state = self.channel_squared_error
        channel_target_state = self.channel_target_squared
        if channel_abs_state is None or channel_sq_state is None or channel_target_state is None:
            channel_abs_state = [0.0] * n_channels
            channel_sq_state = [0.0] * n_channels
            channel_target_state = [0.0] * n_channels
        elif len(channel_abs_state) != n_channels:
            raise ValueError("channel count changed between metric updates")

        error = (pred.detach() - target.detach()).reshape(-1, n_channels)
        target_values = target.detach().reshape(-1, n_channels)
        if mask is not None:
            expected_shape = pred.shape[:-1] if pred.ndim > 1 else pred.shape
            if tuple(mask.shape) != tuple(expected_shape):
                raise ValueError(f"mask must have point shape {tuple(expected_shape)}")
            valid = mask.detach().to(device=pred.device, dtype=torch.bool).reshape(-1)
            error = error[valid]
            target_values = target_values[valid]

        if error.shape[0] == 0:
            return

        error = error.to(dtype=torch.float64)
        target_values = target_values.to(dtype=torch.float64)
        channel_abs = error.abs().sum(dim=0).cpu().tolist()
        channel_squared = error.square().sum(dim=0).cpu().tolist()
        channel_target_squared = target_values.square().sum(dim=0).cpu().tolist()

        self.point_count += error.shape[0]
        self.element_count += error.numel()
        self.total_abs_error += sum(channel_abs)
        self.total_squared_error += sum(channel_squared)
        self.total_target_squared += sum(channel_target_squared)
        self.channel_abs_error = [a + b for a, b in zip(channel_abs_state, channel_abs, strict=True)]
        self.channel_squared_error = [a + b for a, b in zip(channel_sq_state, channel_squared, strict=True)]
        self.channel_target_squared = [
            a + b for a, b in zip(channel_target_state, channel_target_squared, strict=True)
        ]

    @staticmethod
    def _relative_l2(error_squared: float, target_squared: float) -> float:
        epsilon = 1e-12
        return math.sqrt(error_squared) / (math.sqrt(target_squared) + epsilon)

    def compute(self) -> dict[str, MetricValue]:
        if self.element_count == 0:
            return {
                "mae": 0.0,
                "mse": 0.0,
                "rmse": 0.0,
                "relative_l2": 0.0,
                "mae_per_channel": [],
                "rmse_per_channel": [],
                "relative_l2_per_channel": [],
            }

        mse = self.total_squared_error / self.element_count
        assert self.channel_abs_error is not None
        assert self.channel_squared_error is not None
        assert self.channel_target_squared is not None
        return {
            "mae": self.total_abs_error / self.element_count,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "relative_l2": self._relative_l2(
                self.total_squared_error, self.total_target_squared
            ),
            "mae_per_channel": [value / self.point_count for value in self.channel_abs_error],
            "rmse_per_channel": [
                math.sqrt(value / self.point_count) for value in self.channel_squared_error
            ],
            "relative_l2_per_channel": [
                self._relative_l2(error, target)
                for error, target in zip(
                    self.channel_squared_error, self.channel_target_squared, strict=True
                )
            ],
        }


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, MetricValue]:
    """Compute the same aggregate metrics as :class:`MetricsCalculator`."""
    calculator = MetricsCalculator()
    calculator.update(pred, target, mask)
    return calculator.compute()
