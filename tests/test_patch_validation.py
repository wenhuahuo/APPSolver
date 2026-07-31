from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.train_patch import create_datasets, validate
from src.models.patch import LearnedPatchTransformer


def _write_cache(path: Path, offset: float, points_per_side: int):
    path.mkdir(parents=True)
    x, y = np.meshgrid(
        np.linspace(0, 2, points_per_side),
        np.linspace(-1, 1, points_per_side),
    )
    coords0 = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)
    coords = np.repeat(coords0[None], 70, axis=0)
    t = np.arange(70, dtype=np.float32)[:, None, None]
    p = np.arange(len(coords0), dtype=np.float32)[None, :, None] / 100.0
    c = np.arange(4, dtype=np.float32)[None, None, :]
    flows = offset + t + p + c
    np.savez(
        path / 'flow_cache.npz', coords=coords, flows=flows.astype(np.float32),
        channels=np.array(['U:0', 'U:1', 'U:2', 'p_rgh']),
        frame_indices=np.arange(len(coords)),
    )


class _ZeroPatchModel(torch.nn.Module):
    def __init__(self, output_width):
        super().__init__()
        self.output_width = output_width

    def forward(self, inputs, params_embed=None):
        return inputs.new_zeros(inputs.shape[0], inputs.shape[1], self.output_width)


class _MSE:
    def __call__(self, pred, target, mask):
        return (pred - target).square().mean()


def test_partition_mode_disables_distance_refinement(tmp_path):
    case = tmp_path / 'case'
    _write_cache(case, 0.0, 9)

    adaptive, _ = create_datasets(
        dataset_type='ship', data_dirs=[str(case)], patch_size=16,
        output_dim=4, train_ratio=0.8, seed=2, use_embedding=False,
        multi_condition=False, enable_downsample=False,
        rollout_holdout_steps=10, partition_mode='adaptive',
    )
    uniform, _ = create_datasets(
        dataset_type='ship', data_dirs=[str(case)], patch_size=16,
        output_dim=4, train_ratio=0.8, seed=2, use_embedding=False,
        multi_condition=False, enable_downsample=False,
        rollout_holdout_steps=10, partition_mode='uniform',
    )

    assert adaptive.quadtree.enable_distance_refine is True
    assert uniform.quadtree.enable_distance_refine is False
    assert adaptive.num_patches > uniform.num_patches


def test_learned_patch_transformer_shape_and_assignment_gradient():
    model = LearnedPatchTransformer(
        in_flattened_dim=24,
        out_flattened_dim=16,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        slice_num=8,
    )
    inputs = torch.randn(2, 12, 24)
    output = model(inputs)
    output.square().mean().backward()

    assert output.shape == (2, 12, 16)
    assert model.transformer[0].attention.slice_proj.weight.grad is not None


def test_multicondition_validation_scores_all_full_points(tmp_path):
    case_a = tmp_path / 'a'
    case_b = tmp_path / 'b'
    _write_cache(case_a, 0.0, 7)
    _write_cache(case_b, 50.0, 9)

    _, val_dataset = create_datasets(
        dataset_type='ship', data_dirs=[str(case_a), str(case_b)],
        patch_size=16, output_dim=4, train_ratio=0.8, seed=2,
        use_embedding=False, multi_condition=True,
        downsample_method='uniform', downsample_ratio=0.5,
        rollout_holdout_steps=10,
    )
    loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    model = _ZeroPatchModel(val_dataset.global_max_points * 4)

    _, metrics = validate(
        model, loader, _MSE(), torch.device('cpu'), val_dataset,
    )

    all_targets = []
    for batch in loader:
        for target, point_mask in zip(batch['full_target'], batch['full_point_mask']):
            all_targets.append(target[point_mask])
    expected_mae = torch.cat(all_targets).abs().mean().item()

    assert metrics['mae'] == pytest.approx(expected_mae)
    assert len(metrics['mae_per_channel']) == 4
    assert metrics['relative_l2'] == pytest.approx(1.0)
