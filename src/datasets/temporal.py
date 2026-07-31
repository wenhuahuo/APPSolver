"""Temporal split and training-only normalization utilities."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    test: np.ndarray
    rollout_frames: np.ndarray


def stable_condition_seed(seed: int, identity: str) -> int:
    """Derive an order-independent split seed from a condition path."""
    stable_name = '/'.join(Path(identity).parts[-3:])
    digest = hashlib.sha256(stable_name.encode('utf-8')).digest()
    offset = int.from_bytes(digest[:4], byteorder='little')
    return (int(seed) + offset) % (2**32)


def make_temporal_split(
    n_timesteps: int,
    step_size: int,
    train_ratio: float,
    seed: int,
    rollout_steps: int = 0,
) -> TemporalSplit:
    """Build true temporal pairs, then randomly split pair start indices.

    A contiguous rollout window can be reserved first. Any pair touching a
    rollout frame is excluded from both the one-step train and test pools.
    """
    if step_size < 1:
        raise ValueError("step_size must be >= 1")
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be in [0, 1]")

    pair_starts = np.arange(max(0, n_timesteps - step_size), dtype=np.int64)
    rollout_frames = np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)

    if rollout_steps > 0:
        if step_size != 1:
            raise ValueError("rollout holdout currently requires one-step temporal pairs")
        last_start = n_timesteps - 1 - rollout_steps
        if last_start < 0:
            raise ValueError(
                f"{n_timesteps} timesteps are insufficient for a "
                f"{rollout_steps}-step rollout with step_size={step_size}"
            )
        rollout_start = int(rng.integers(0, last_start + 1))
        rollout_frames = rollout_start + np.arange(
            rollout_steps + 1, dtype=np.int64
        )
        touches_rollout = np.isin(pair_starts, rollout_frames) | np.isin(
            pair_starts + step_size, rollout_frames
        )
        pair_starts = pair_starts[~touches_rollout]

    shuffled = rng.permutation(pair_starts)
    n_train = int(len(shuffled) * train_ratio)
    train = np.sort(shuffled[:n_train])
    test = np.sort(shuffled[n_train:])
    return TemporalSplit(train=train, test=test, rollout_frames=rollout_frames)


def frame_indices_for_pairs(pair_indices: np.ndarray, step_size: int) -> np.ndarray:
    if len(pair_indices) == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate([pair_indices, pair_indices + step_size]))


def compute_global_normalization_params(datasets: Iterable[object]) -> Dict[str, np.ndarray]:
    """Compute coordinate/flow Z-score parameters from training pairs only."""
    coord_sum = coord_sq_sum = flow_sum = flow_sq_sum = None
    coord_count = flow_count = 0

    for dataset in datasets:
        frame_indices = frame_indices_for_pairs(
            dataset.temporal_split.train, dataset.step_size
        )
        if len(frame_indices) == 0:
            continue

        for frame_idx in frame_indices:
            coords = np.asarray(dataset._all_coords[frame_idx], dtype=np.float64)
            flows = np.asarray(dataset._all_flows[frame_idx], dtype=np.float64)

            if coord_sum is None:
                coord_sum = np.zeros(coords.shape[-1], dtype=np.float64)
                coord_sq_sum = np.zeros_like(coord_sum)
                flow_sum = np.zeros(flows.shape[-1], dtype=np.float64)
                flow_sq_sum = np.zeros_like(flow_sum)

            coord_sum += coords.sum(axis=0)
            coord_sq_sum += np.square(coords).sum(axis=0)
            flow_sum += flows.sum(axis=0)
            flow_sq_sum += np.square(flows).sum(axis=0)
            coord_count += coords.shape[0]
            flow_count += flows.shape[0]

    if coord_count == 0 or flow_count == 0:
        raise ValueError("Cannot compute normalization statistics from an empty training split")

    coord_mean = coord_sum / coord_count
    flow_mean = flow_sum / flow_count
    coord_var = np.maximum(coord_sq_sum / coord_count - np.square(coord_mean), 0.0)
    flow_var = np.maximum(flow_sq_sum / flow_count - np.square(flow_mean), 0.0)

    return {
        'coord_mean': coord_mean.astype(np.float32),
        'coord_std': (np.sqrt(coord_var) + 1e-8).astype(np.float32),
        'flow_mean': flow_mean.astype(np.float32),
        'flow_std': (np.sqrt(flow_var) + 1e-8).astype(np.float32),
    }


def copy_normalization_params(params: Optional[Dict[str, np.ndarray]]):
    if params is None:
        return None
    return {key: np.asarray(value, dtype=np.float32).copy() for key, value in params.items()}


def _dataset_conditions(dataset):
    return dataset.sub_datasets if hasattr(dataset, 'sub_datasets') else [dataset]


def save_data_protocol(output_dir: str, train_dataset, val_dataset) -> None:
    """Persist normalization statistics and temporal indices with a run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    params = train_dataset.get_normalization_params()
    np.savez(output_path / 'normalization_stats.npz', **params)
    if hasattr(train_dataset, 'get_condition_normalization_params'):
        condition_params = train_dataset.get_condition_normalization_params()
        if condition_params is not None:
            np.savez(
                output_path / 'condition_normalization_stats.npz',
                **condition_params,
            )

    def describe(dataset):
        conditions = []
        for condition in _dataset_conditions(dataset):
            name = getattr(condition, 'data_dir', None)
            if name is None:
                name = f"{condition.root}/{condition.benchmark}/{condition.case}"
            conditions.append({
                'name': str(name),
                'split': condition.split,
                'split_seed': int(condition.split_seed),
                'pair_indices': condition.pair_indices.tolist(),
                'rollout_frames': condition.temporal_split.rollout_frames.tolist(),
            })
        return conditions

    manifest = {
        'train': describe(train_dataset),
        'validation': describe(val_dataset),
    }
    with open(output_path / 'data_protocol.json', 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
