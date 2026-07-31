from pathlib import Path

import numpy as np

from scripts.train_patch import create_datasets
from src.datasets.cfdBench import CFDBenchPatchDataset


def test_cfdbench_patch_target_contains_only_flow_channels(tmp_path: Path):
    data_dir = tmp_path / '01_cavityflow' / 'case0'
    data_dir.mkdir(parents=True)
    x, y = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6))
    coords0 = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)
    coords = np.repeat(coords0[None], 20, axis=0)
    t = np.arange(20, dtype=np.float32)[:, None, None]
    flows = np.concatenate([
        np.broadcast_to(t, (20, len(coords0), 1)),
        np.broadcast_to(2 * t, (20, len(coords0), 1)),
    ], axis=-1)
    np.savez(
        data_dir / 'flow_cache.npz', coords=coords, flows=flows,
        channels=np.array(['y-velocity', 'x-velocity']),
    )

    dataset = CFDBenchPatchDataset(
        root=str(tmp_path), benchmark='01_cavityflow', case='case0',
        patch_size=8, downsample_ratio=0.5,
    )
    sample = dataset[0]

    assert sample['output'].shape[-1] == dataset.max_points * dataset.output_dim
    assert sample['full_target'].shape == (dataset.num_points, dataset.output_dim)


def test_multicondition_cfdbench_patch_dataset_reuses_training_normalization(
    tmp_path: Path,
):
    case_dirs = []
    for case_id, point_count in enumerate((36, 49)):
        case_dir = tmp_path / '01_cavityflow' / f'case{case_id}'
        case_dir.mkdir(parents=True)
        side = int(np.sqrt(point_count))
        x, y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        base = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)
        coords = np.repeat(base[None], 12, axis=0)
        time = np.arange(12, dtype=np.float32)[:, None, None]
        flows = np.concatenate([
            np.broadcast_to(time + case_id, (12, len(base), 1)),
            np.broadcast_to(2 * time + case_id, (12, len(base), 1)),
        ], axis=-1)
        np.savez(
            case_dir / 'flow_cache.npz',
            coords=coords,
            flows=flows,
            channels=np.array(['y-velocity', 'x-velocity']),
        )
        case_dirs.append(str(case_dir))

    train, validation = create_datasets(
        dataset_type='cfd_bench',
        data_dirs=case_dirs,
        patch_size=8,
        output_dim=2,
        train_ratio=0.8,
        seed=42,
        use_embedding=False,
        multi_condition=True,
        enable_downsample=False,
        rollout_holdout_steps=0,
        partition_mode='uniform',
    )

    assert train.num_conditions == validation.num_conditions == 2
    for train_case, validation_case in zip(
        train.sub_datasets, validation.sub_datasets,
    ):
        np.testing.assert_array_equal(train_case.flow_mean, validation_case.flow_mean)
        np.testing.assert_array_equal(train_case.flow_std, validation_case.flow_std)
