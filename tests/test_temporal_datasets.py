from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.samplers import ConditionBatchSampler
from src.datasets.temporal import make_temporal_split
from src.datasets.shipBench import (
    IrregularFlowFieldDataset,
    MultiConditionIrregularDataset,
    PatchFlowFieldDataset,
)
from src.core.metrics import patches_to_points, recover_points_knn


def _write_ship_cache(path: Path, timesteps=80, nx=8, ny=8, offset=0.0):
    path.mkdir(parents=True)
    x, y = np.meshgrid(np.linspace(0, 2, nx), np.linspace(-1, 1, ny))
    base_coords = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)
    coords = np.repeat(base_coords[None], timesteps, axis=0)
    time = np.arange(timesteps, dtype=np.float32)[:, None, None]
    point = np.arange(len(base_coords), dtype=np.float32)[None, :, None] / 100.0
    channels = np.arange(4, dtype=np.float32)[None, None, :]
    flows = offset + time + point + channels
    np.savez(
        path / 'flow_cache.npz',
        coords=coords,
        flows=flows.astype(np.float32),
        channels=np.array(['U:0', 'U:1', 'U:2', 'p_rgh']),
        frame_indices=np.arange(timesteps),
    )


def test_rollout_holdout_requires_one_step_pairs():
    with pytest.raises(ValueError, match='one-step'):
        make_temporal_split(100, step_size=2, train_ratio=0.8, seed=1, rollout_steps=10)


def test_random_pair_split_preserves_transitions_and_reserves_rollout(tmp_path):
    data_dir = tmp_path / 'case'
    _write_ship_cache(data_dir)
    dataset = IrregularFlowFieldDataset(
        str(data_dir), train_ratio=0.8, seed=7, rollout_holdout_steps=10,
    )

    train = set(dataset.temporal_split.train.tolist())
    test = set(dataset.temporal_split.test.tolist())
    rollout = set(dataset.temporal_split.rollout_frames.tolist())

    assert train.isdisjoint(test)
    assert len(dataset.temporal_split.rollout_frames) == 11
    assert all(t not in rollout and t + 1 not in rollout for t in train | test)

    time_idx = int(dataset.pair_indices[0])
    _, current, target = dataset[0]
    current_raw = current.numpy() * dataset.flow_std + dataset.flow_mean
    target_raw = target.numpy() * dataset.flow_std + dataset.flow_mean
    assert np.allclose(current_raw, dataset._all_flows[time_idx], atol=1e-5)
    assert np.allclose(target_raw, dataset._all_flows[time_idx + 1], atol=1e-5)


def test_multi_condition_uses_one_global_training_normalizer_and_full_points(tmp_path):
    case_a = tmp_path / 'a'
    case_b = tmp_path / 'b'
    _write_ship_cache(case_a, nx=6, ny=6, offset=0.0)
    _write_ship_cache(case_b, nx=8, ny=8, offset=100.0)

    dataset = MultiConditionIrregularDataset(
        [str(case_a), str(case_b)], rollout_holdout_steps=10, seed=3,
    )
    params = dataset.get_normalization_params()

    assert dataset.global_max_points == 64
    assert dataset[0][0].shape[0] in {36, 64}
    for sub_dataset in dataset.sub_datasets:
        assert np.array_equal(sub_dataset.flow_mean, params['flow_mean'])
        assert np.array_equal(sub_dataset.flow_std, params['flow_std'])

    standalone = IrregularFlowFieldDataset(
        str(case_b), rollout_holdout_steps=10, seed=3,
    )
    assert np.array_equal(
        standalone.temporal_split.rollout_frames,
        dataset.sub_datasets[1].temporal_split.rollout_frames,
    )
    assert np.array_equal(
        standalone.temporal_split.train,
        dataset.sub_datasets[1].temporal_split.train,
    )

    for batch in ConditionBatchSampler(dataset, batch_size=4, shuffle=False):
        condition_ids = {dataset._index_map[index][0] for index in batch}
        assert len(condition_ids) == 1


def test_patch_dataset_recovers_prediction_to_every_original_point(tmp_path):
    data_dir = tmp_path / 'patch_case'
    _write_ship_cache(data_dir, timesteps=70, nx=10, ny=10)
    dataset = PatchFlowFieldDataset(
        str(data_dir), patch_size=16, downsample_ratio=0.5,
        downsample_method='uniform', rollout_holdout_steps=10, split='test',
    )
    sample = dataset[0]

    assert sample['full_target'].shape == (dataset.num_points, 4)
    assert torch.equal(sample['mask'], torch.from_numpy(dataset._patch_mask))
    input_values = sample['input'].reshape(
        dataset.num_patches, dataset.max_points, dataset.input_dim
    )
    time_idx = int(dataset.pair_indices[0])
    expected_values = np.concatenate([
        dataset._normalize_coords(dataset._all_coords[time_idx]),
        dataset._normalize_flows(dataset._all_flows[time_idx]),
    ], axis=1)
    assert np.allclose(
        input_values.numpy()[dataset._patch_mask],
        expected_values[dataset._patch_indices[dataset._patch_mask]],
    )
    assert np.all(input_values.numpy()[~dataset._patch_mask] == 0)
    sampled_points, sampled_mask = patches_to_points(
        sample['output'].unsqueeze(0), dataset.quadtree, 1,
        dataset.num_points, dataset.output_dim, dataset.output_dim,
        dataset.max_points,
    )
    recovered = recover_points_knn(
        sampled_points, dataset.recovery_indices, dataset.recovery_weights,
        dataset.sampled_indices,
    )

    assert recovered.shape == (1, dataset.num_points, dataset.output_dim)
    assert sampled_mask.sum().item() == len(dataset.sampled_indices)
    assert torch.equal(
        recovered[0, dataset.sampled_indices],
        sampled_points[0, dataset.sampled_indices],
    )
