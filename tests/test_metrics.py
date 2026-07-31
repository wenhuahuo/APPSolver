import math
from types import SimpleNamespace

import pytest
import torch

from src.core.metrics import MetricsCalculator, patches_to_points, recover_points_knn


def test_patches_to_points_keeps_existing_layout_and_mask():
    tree = SimpleNamespace(
        patches=[SimpleNamespace(points=[0, 2]), SimpleNamespace(points=[1])]
    )
    patches = torch.tensor([[[1.0, 10.0, 2.0, 20.0], [3.0, 30.0, 0.0, 0.0]]])

    points, mask = patches_to_points(patches, tree, 1, 3, 1, 2, 2)

    assert points[:, :, 0].tolist() == [[1.0, 3.0, 2.0]]
    assert mask.tolist() == [[True, True, True]]


def test_knn_recovery_is_differentiable_and_keeps_samples_exact():
    sampled = torch.tensor([[10.0], [40.0]], requires_grad=True)
    sampled_indices = torch.tensor([0, 3])
    neighbors = torch.tensor([[0, 3], [0, 3], [0, 3], [0, 3]])
    weights = torch.tensor(
        [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.5, 0.5]],
        requires_grad=True,
    )

    recovered = recover_points_knn(sampled, neighbors, weights, sampled_indices)

    assert recovered[:, 0].tolist() == pytest.approx([10.0, 17.5, 32.5, 40.0])
    assert torch.equal(recovered[sampled_indices], sampled)
    recovered.sum().backward()
    assert sampled.grad is not None
    assert weights.grad is not None


def test_metrics_aggregate_elements_apply_mask_and_report_channels():
    calculator = MetricsCalculator()
    calculator.update(
        pred=torch.tensor([[[2.0, 4.0], [100.0, 100.0]]]),
        target=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        mask=torch.tensor([[True, False]]),
    )
    calculator.update(
        pred=torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
        target=torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
    )

    metrics = calculator.compute()

    assert metrics["mae"] == pytest.approx(3.0 / 6.0)
    assert metrics["mse"] == pytest.approx(5.0 / 6.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(5.0 / 6.0))
    assert metrics["relative_l2"] == pytest.approx(math.sqrt(5.0 / 15.0))
    assert metrics["mae_per_channel"] == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert metrics["rmse_per_channel"] == pytest.approx(
        [math.sqrt(1.0 / 3.0), math.sqrt(4.0 / 3.0)]
    )
    assert metrics["relative_l2_per_channel"] == pytest.approx(
        [1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
    )

    # Existing patch evaluation passes boolean-indexed (flattened) tensors.
    flattened = MetricsCalculator()
    flattened.update(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))
    assert flattened.compute()["mae"] == pytest.approx(1.5)
