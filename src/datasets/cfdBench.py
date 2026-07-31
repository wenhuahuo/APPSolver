"""
CFDBench Dataset Loader

Supports loading CFD Bench data (03_damflow, 04_cylinderflow, etc.)
with both irregular and patch formats aligned with ship.py interface.

Data format per txt file:
    nodenumber  x-coordinate  y-coordinate  water-vof  y-velocity  x-velocity
"""

import os
import copy
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import KDTree

from .base import BaseDataset
from .temporal import (
    compute_global_normalization_params,
    copy_normalization_params,
    make_temporal_split,
    stable_condition_seed,
)
from ..data_processor.mesh_quad import QuadTreeMesh


def _clone_dataset_with_split(dataset: Dataset, split: str) -> Dataset:
    cloned = copy.copy(dataset)
    cloned.set_split(split)
    return cloned


def _load_npz_cache(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    if not os.path.exists(cache_path):
        return None

    data = np.load(cache_path, allow_pickle=True)
    if 'coords' not in data or 'flows' not in data or 'channels' not in data:
        return None

    coords = data['coords'].astype(np.float32)
    flows = data['flows'].astype(np.float32)
    channels = [str(c) for c in data['channels'].tolist()]
    return coords, flows, channels


CFD_COLUMN_ALIASES = {
    'volume-fraction-water': 'water-vof',
    'y-velocity-water': 'y-velocity',
    'x-velocity-water': 'x-velocity',
}


def _canonicalize_cfd_channel_name(name: str) -> str:
    return CFD_COLUMN_ALIASES.get(name, name)


def _canonicalize_cfd_channels(names: List[str]) -> List[str]:
    return [_canonicalize_cfd_channel_name(name) for name in names]


def _standardize_cfd_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for alias, canonical in CFD_COLUMN_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            rename_map[alias] = canonical

    if rename_map:
        return df.rename(columns=rename_map)
    return df


class CFDBenchIrregularDataset(Dataset):
    """
    CFDBench Irregular format dataset

    Output format:
        pos: (N, 2) - x, y coordinates
        fx: (N, C) - flow features at time t
        y: (N, C) - flow features at time t+1
    """

    BENCHMARK_TYPES = ['01_cavityflow', '02_tubeflow', '03_damflow', '04_cylinderflow']
    FIELD_COLS = ['x-coordinate', 'y-coordinate', 'y-velocity', 'x-velocity']

    def __init__(
        self,
        root: str,
        benchmark: str = '03_damflow',
        case: str = 'case0',
        step_size: int = 1,
        train_ratio: float = 0.8,
        seed: int = 42,
        output_channels: List[str] = None,
        normalize: bool = True,
        prefer_cache: bool = True,
        cache_filename: str = 'flow_cache.npz',
        rollout_holdout_steps: int = 0,
        normalization_params: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.root = root
        self.benchmark = benchmark
        self.case = case
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.seed = seed
        self.split_seed = stable_condition_seed(
            seed, os.path.join(root, benchmark, case)
        )
        self.output_channels = _canonicalize_cfd_channels(
            output_channels or ['y-velocity', 'x-velocity']
        )
        self.normalize = normalize
        self.prefer_cache = prefer_cache
        self.cache_filename = cache_filename
        self.rollout_holdout_steps = rollout_holdout_steps
        self._normalization_params = copy_normalization_params(normalization_params)

        self.split = 'train'
        self._already_split = False

        self._load_data()
        self._split_data()
        self._compute_normalization_params()

    def _load_data(self):
        data_dir = os.path.join(self.root, self.benchmark, self.case)

        if self.prefer_cache:
            cached = _load_npz_cache(os.path.join(data_dir, self.cache_filename))
            if cached is not None:
                coords, all_flows, channel_names = cached
                channel_names = _canonicalize_cfd_channels(channel_names)
                channel_to_idx = {name: i for i, name in enumerate(channel_names)}
                flow_cols = [c for c in self.output_channels if c in channel_to_idx]
                if not flow_cols:
                    raise ValueError(
                        f"No requested output channels in cache: {self.output_channels}; "
                        f"available: {channel_names}"
                    )

                indices = [channel_to_idx[c] for c in flow_cols]
                self.coords = coords
                self.flows = all_flows[:, :, indices]
                self.n_timesteps = len(self.coords)
                self.n_points = self.coords.shape[1]
                self.n_channels = self.flows.shape[2]
                return

        data_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith('data') and f.endswith('.txt')
        ])

        coords_list = []
        flows_list = []

        for data_file in data_files:
            data_path = os.path.join(data_dir, data_file)
            df = pd.read_csv(data_path, sep=r'\s+')
            df = _standardize_cfd_columns(df)

            coords = df[['x-coordinate', 'y-coordinate']].values.astype(np.float32)

            flow_cols = [c for c in self.output_channels if c in df.columns]
            if not flow_cols:
                raise ValueError(
                    f"No requested output channels in file: {data_path}; "
                    f"requested: {self.output_channels}; available: {list(df.columns)}"
                )
            flows = df[flow_cols].values.astype(np.float32)

            coords_list.append(coords)
            flows_list.append(flows)

        min_cells = min(len(c) for c in coords_list)
        reference_coords = coords_list[0]

        sorted_coords = []
        sorted_flows = []

        for coords, flows in zip(coords_list, flows_list):
            tree = KDTree(coords)
            _distance, indices = tree.query(reference_coords[:min_cells])
            sorted_coords.append(coords[indices])
            sorted_flows.append(flows[indices])

        self.coords = np.array(sorted_coords, dtype=np.float32)
        self.flows = np.array(sorted_flows, dtype=np.float32)

        self.n_timesteps = len(self.coords)
        self.n_points = min_cells
        self.n_channels = self.flows.shape[2]

    def _split_data(self):
        if not hasattr(self, '_all_coords'):
            self._all_coords = self.coords
            self._all_flows = self.flows
            self._original_n_timesteps = self.n_timesteps
            self.temporal_split = make_temporal_split(
                self.n_timesteps,
                self.step_size,
                self.train_ratio,
                self.split_seed,
                self.rollout_holdout_steps,
            )

        if self.split == 'train':
            self.pair_indices = self.temporal_split.train
        elif self.split == 'test':
            self.pair_indices = self.temporal_split.test
        elif self.split == 'all':
            self.pair_indices = np.arange(
                max(0, self._original_n_timesteps - self.step_size), dtype=np.int64
            )
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.coords = self._all_coords
        self.flows = self._all_flows
        self.n_timesteps = self._original_n_timesteps
        self._already_split = True

    def set_split(self, split: str):
        self.split = split
        self._already_split = False
        self._split_data()

    def set_normalization_params(self, params: Dict[str, np.ndarray]) -> None:
        self._normalization_params = copy_normalization_params(params)
        self._compute_normalization_params()

    def _compute_normalization_params(self):
        if not self.normalize:
            self.coord_mean = self.coord_std = None
            self.flow_mean = self.flow_std = None
            return

        params = self._normalization_params
        if params is None:
            params = compute_global_normalization_params([self])
        self.coord_mean = params['coord_mean']
        self.coord_std = params['coord_std']
        self.flow_mean = params['flow_mean']
        self.flow_std = params['flow_std']

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return coords
        return (coords - self.coord_mean) / self.coord_std

    def _normalize_flows(self, flows: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return flows
        return (flows - self.flow_mean) / self.flow_std

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_idx = int(self.pair_indices[idx])
        coords_t = self._all_coords[time_idx]
        flow_t = self._all_flows[time_idx]
        flow_tp1 = self._all_flows[time_idx + self.step_size]

        pos = torch.from_numpy(self._normalize_coords(coords_t)).float()
        fx = torch.from_numpy(self._normalize_flows(flow_t)).float()
        y = torch.from_numpy(self._normalize_flows(flow_tp1)).float()

        return pos, fx, y

    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        return {
            'coord_mean': self.coord_mean,
            'coord_std': self.coord_std,
            'flow_mean': self.flow_mean,
            'flow_std': self.flow_std,
        }

    def get_rollout_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        indices = self.temporal_split.rollout_frames
        return self._all_coords[indices], self._all_flows[indices]


class CFDBenchPatchDataset(Dataset):
    """
    CFDBench Patch format dataset

    Output format:
        input: (P, N*C) flattened patches
        output: (P, N*C) flattened patches
        mask: (P, N) valid point mask
    """

    BENCHMARK_TYPES = ['01_cavityflow', '02_tubeflow', '03_damflow', '04_cylinderflow']
    FIELD_COLS = ['x-coordinate', 'y-coordinate', 'y-velocity', 'x-velocity']

    def __init__(
        self,
        root: str,
        benchmark: str = '03_damflow',
        case: str = 'case0',
        step_size: int = 1,
        patch_size: int = 64,
        enable_downsample: bool = True,
        downsample_method: str = 'uniform',
        downsample_ratio: float = 0.25,
        include_coordinates: bool = True,
        output_dim: int = 2,
        normalize: bool = True,
        split: str = 'train',
        train_ratio: float = 0.8,
        seed: int = 42,
        prefer_cache: bool = True,
        cache_filename: str = 'flow_cache.npz',
        rollout_holdout_steps: int = 0,
        normalization_params: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.root = root
        self.benchmark = benchmark
        self.case = case
        self.step_size = step_size
        self.patch_size = patch_size
        self.enable_downsample = enable_downsample
        self.downsample_method = downsample_method
        self.downsample_ratio = downsample_ratio
        self.include_coordinates = include_coordinates
        self.output_dim = output_dim
        self.normalize = normalize
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.split_seed = stable_condition_seed(
            seed, os.path.join(root, benchmark, case)
        )
        self.prefer_cache = prefer_cache
        self.cache_filename = cache_filename
        self.rollout_holdout_steps = rollout_holdout_steps
        self._normalization_params = copy_normalization_params(normalization_params)

        self._load_data()
        self._split_data()
        self._build_quadtree()
        self._compute_normalization_params()

    def _load_data(self):
        data_dir = os.path.join(self.root, self.benchmark, self.case)

        if self.prefer_cache:
            cached = _load_npz_cache(os.path.join(data_dir, self.cache_filename))
            if cached is not None:
                coords, all_flows, channel_names = cached
                channel_names = _canonicalize_cfd_channels(channel_names)
                channel_to_idx = {name: i for i, name in enumerate(channel_names)}
                required = ['y-velocity', 'x-velocity']
                if not all(ch in channel_to_idx for ch in required[:self.output_dim]):
                    raise ValueError(
                        f"Cache missing required CFD channels for output_dim={self.output_dim}; "
                        f"available: {channel_names}"
                    )

                sel_idx = [channel_to_idx[ch] for ch in required[:self.output_dim]]
                self.coords = coords
                self.flows = all_flows[:, :, sel_idx]
                self.num_timesteps = len(self.coords)
                self.num_points = self.coords.shape[1]
                self.output_dim = self.flows.shape[2]
                return

        data_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith('data') and f.endswith('.txt')
        ])

        coords_list = []
        flows_list = []

        for data_file in data_files:
            data_path = os.path.join(data_dir, data_file)
            df = pd.read_csv(data_path, sep=r'\s+')
            df = _standardize_cfd_columns(df)

            coords = df[['x-coordinate', 'y-coordinate']].values.astype(np.float32)

            flow_values = df[['y-velocity', 'x-velocity']].values.astype(np.float32)
            flow_values = flow_values[:, :self.output_dim]

            coords_list.append(coords)
            flows_list.append(flow_values)

        min_cells = min(len(c) for c in coords_list)
        reference_coords = coords_list[0]

        sorted_coords = []
        sorted_flows = []

        for coords, flows in zip(coords_list, flows_list):
            tree = KDTree(coords)
            _distance, indices = tree.query(reference_coords[:min_cells])
            sorted_coords.append(coords[indices])
            sorted_flows.append(flows[indices])

        self.coords = np.array(sorted_coords, dtype=np.float32)
        self.flows = np.array(sorted_flows, dtype=np.float32)

        self.num_timesteps = len(self.coords)
        self.num_points = min_cells
        self.output_dim = self.flows.shape[2]

    def _split_data(self):
        if not hasattr(self, '_all_coords'):
            self._all_coords = self.coords
            self._all_flows = self.flows
            self._original_n_timesteps = self.num_timesteps
            self.temporal_split = make_temporal_split(
                self.num_timesteps,
                self.step_size,
                self.train_ratio,
                self.split_seed,
                self.rollout_holdout_steps,
            )

        if self.split == 'train':
            self.pair_indices = self.temporal_split.train
        elif self.split == 'test':
            self.pair_indices = self.temporal_split.test
        elif self.split == 'all':
            self.pair_indices = np.arange(
                max(0, self._original_n_timesteps - self.step_size), dtype=np.int64
            )
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.coords = self._all_coords
        self.flows = self._all_flows
        self.num_timesteps = self._original_n_timesteps
        self._already_split = True

    def set_split(self, split: str):
        self.split = split
        self._already_split = False
        self._split_data()

    def _build_quadtree(self):
        self.quadtree = QuadTreeMesh(
            self.coords[0],
            patch_size=self.patch_size,
            enable_distance_refine=False,
        )

        if self.enable_downsample:
            target_points = max(4, int(self.patch_size * self.downsample_ratio))
            self.quadtree.downsample_patches_by_distance(
                method=self.downsample_method,
                target_points=target_points,
                min_points=4
            )

        self.num_patches = len(self.quadtree.patches)
        self.max_points = max(len(p.points) for p in self.quadtree.patches)
        self._build_recovery_map()

        self.coord_dim = 2 if self.include_coordinates else 0
        self.input_dim = self.coord_dim + self.output_dim

    def _build_recovery_map(self) -> None:
        sampled = np.concatenate([patch.points for patch in self.quadtree.patches])
        sampled = np.unique(sampled).astype(np.int64)
        if len(sampled) < 4:
            raise ValueError("APP recovery requires at least four sampled points")

        full_coords = self._all_coords[0]
        tree = KDTree(full_coords[sampled])
        distances, compact_neighbors = tree.query(full_coords, k=4)
        weights = 1.0 / np.maximum(distances, 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)

        self.sampled_indices = torch.from_numpy(sampled).long()
        self.recovery_indices = torch.from_numpy(sampled[compact_neighbors]).long()
        self.recovery_weights = torch.from_numpy(weights.astype(np.float32))

    def set_normalization_params(self, params: Dict[str, np.ndarray]) -> None:
        self._normalization_params = copy_normalization_params(params)
        self._compute_normalization_params()

    def _compute_normalization_params(self):
        if not self.normalize:
            self.coord_mean = self.coord_std = None
            self.flow_mean = self.flow_std = None
            return

        params = self._normalization_params
        if params is None:
            params = compute_global_normalization_params([self])
        self.coord_mean = params['coord_mean']
        self.coord_std = params['coord_std']
        self.flow_mean = params['flow_mean']
        self.flow_std = params['flow_std']

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return coords
        return (coords - self.coord_mean) / self.coord_std

    def _normalize_flows(self, flows: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return flows
        return (flows - self.flow_mean) / self.flow_std

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        time_idx = int(self.pair_indices[idx])
        input_flows = self._all_flows[time_idx]
        output_flows = self._all_flows[time_idx + self.step_size]

        coords = self._all_coords[time_idx]
        coords_norm = self._normalize_coords(coords)
        input_flows_norm = self._normalize_flows(input_flows)
        output_flows_norm = self._normalize_flows(output_flows)

        input_patches = self._extract_patches(coords_norm, input_flows_norm)
        output_patches = self._extract_flow_patches(output_flows_norm)

        mask = self._create_mask()

        return {
            'input': input_patches,
            'output': output_patches,
            'mask': mask,
            'full_target': torch.from_numpy(output_flows_norm).float(),
            'time_index': torch.tensor(time_idx, dtype=torch.long),
        }

    def _extract_patches(self, coords: np.ndarray, flows: np.ndarray) -> torch.Tensor:
        patches = torch.zeros(
            (self.num_patches, self.max_points * self.input_dim),
            dtype=torch.float32
        )

        for patch_idx, patch in enumerate(self.quadtree.patches):
            point_indices = patch.points
            n_points = len(point_indices)

            patch_coords = coords[point_indices]
            patch_flows = flows[point_indices]

            if self.include_coordinates:
                combined = np.concatenate([patch_coords, patch_flows], axis=1)
            else:
                combined = patch_flows

            flat_combined = combined.flatten()
            patches[patch_idx, :n_points * self.input_dim] = torch.from_numpy(flat_combined)

        return patches

    def _extract_flow_patches(self, flows: np.ndarray) -> torch.Tensor:
        patches = torch.zeros(
            (self.num_patches, self.max_points * self.output_dim),
            dtype=torch.float32,
        )
        for patch_idx, patch in enumerate(self.quadtree.patches):
            point_indices = patch.points
            flat_flows = flows[point_indices].flatten()
            patches[patch_idx, :len(point_indices) * self.output_dim] = torch.from_numpy(
                flat_flows
            )
        return patches

    def _create_mask(self) -> torch.Tensor:
        mask = torch.zeros(
            (self.num_patches, self.max_points),
            dtype=torch.bool
        )

        for patch_idx, patch in enumerate(self.quadtree.patches):
            n_points = len(patch.points)
            mask[patch_idx, :n_points] = True

        return mask

    def get_quadtree(self) -> QuadTreeMesh:
        return self.quadtree

    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        return {
            'coord_mean': self.coord_mean,
            'coord_std': self.coord_std,
            'flow_mean': self.flow_mean,
            'flow_std': self.flow_std,
        }

    def get_rollout_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        indices = self.temporal_split.rollout_frames
        return self._all_coords[indices], self._all_flows[indices]


def create_cfd_bench_irregular_dataloader(
    root: str,
    benchmark: str = '03_damflow',
    case: str = 'case0',
    split: str = 'train',
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    step_size: int = 1,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DataLoader:
    dataset = CFDBenchIrregularDataset(
        root=root,
        benchmark=benchmark,
        case=case,
        step_size=step_size,
        train_ratio=train_ratio,
        seed=seed,
    )
    dataset.set_split(split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_cfd_bench_patch_dataloader(
    root: str,
    benchmark: str = '03_damflow',
    case: str = 'case0',
    split: str = 'train',
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    step_size: int = 1,
    patch_size: int = 64,
    output_dim: int = 2,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DataLoader:
    dataset = CFDBenchPatchDataset(
        root=root,
        benchmark=benchmark,
        case=case,
        step_size=step_size,
        patch_size=patch_size,
        output_dim=output_dim,
        split=split,
        train_ratio=train_ratio,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == '__main__':
    import sys

    root = 'datasets/cfdBench'

    print("=== Testing CFDBenchIrregularDataset ===")
    dataset = CFDBenchIrregularDataset(
        root=root,
        benchmark='03_damflow',
        case='case0',
        step_size=1,
        train_ratio=0.8,
        seed=42,
    )
    dataset.set_split('train')

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Coords shape: {dataset.coords.shape}")
    print(f"Flows shape: {dataset.flows.shape}")

    pos, fx, y = dataset[0]
    print(f"Pos shape: {pos.shape}, FX shape: {fx.shape}, Y shape: {y.shape}")

    dataset.set_split('test')
    print(f"Test set: {len(dataset)} samples")

    print("\n=== Testing CFDBenchPatchDataset ===")
    patch_dataset = CFDBenchPatchDataset(
        root=root,
        benchmark='03_damflow',
        case='case0',
        step_size=1,
        patch_size=64,
        output_dim=2,
        train_ratio=0.8,
        seed=42,
    )
    patch_dataset.set_split('train')

    print(f"Dataset loaded: {len(patch_dataset)} samples")
    print(f"Num patches: {patch_dataset.num_patches}, Max points: {patch_dataset.max_points}")
    print(f"Input dim: {patch_dataset.input_dim}")

    sample = patch_dataset[0]
    print(f"Sample input shape: {sample['input'].shape}")
    print(f"Sample output shape: {sample['output'].shape}")
    print(f"Sample mask shape: {sample['mask'].shape}")

    print("\nAll tests passed!")


class MultiConditionCFDBenchIrregularDataset(Dataset):
    """
    多工况CFDBench Irregular流场数据集

    支持同时加载多个CFD工况目录的数据:
    1. 保留每个工况的完整点集，由 ConditionBatchSampler 按工况组 batch
    2. 不同工况的时间样本通过 (condition_id, local_id) 双索引组织

    输出格式与CFDBenchIrregularDataset一致,额外返回condition_id

    Args:
        split: 'train' 或 'test'，控制每个子数据集使用哪个划分
        max_points: 仅用于记录全局最大点数；数据不会截断
    """

    def __init__(
        self,
        roots: List[str],
        benchmarks: List[str],
        cases: List[str],
        step_size: int = 1,
        train_ratio: float = 0.8,
        seed: int = 42,
        output_channels: List[str] = None,
        normalize: bool = True,
        split: str = 'train',
        max_points: Optional[int] = None,
        rollout_holdout_steps: int = 0,
        normalization_params: Optional[Dict[str, np.ndarray]] = None,
    ):
        assert len(roots) == len(benchmarks) == len(cases), \
            "roots, benchmarks, cases must have the same length"

        self.num_conditions = len(roots)

        shared_kwargs = dict(
            step_size=step_size,
            train_ratio=train_ratio,
            output_channels=output_channels,
            normalize=normalize,
            rollout_holdout_steps=rollout_holdout_steps,
            normalization_params=normalization_params,
        )

        self.sub_datasets: List[CFDBenchIrregularDataset] = []
        for i, (root, benchmark, case) in enumerate(zip(roots, benchmarks, cases)):
            print(f"加载工况 {i} [{split}]: {benchmark}/{case}")
            ds = CFDBenchIrregularDataset(
                root=root, benchmark=benchmark, case=case,
                seed=seed, **shared_kwargs
            )
            ds.set_split(split)
            self.sub_datasets.append(ds)

        if normalize and normalization_params is None:
            normalization_params = compute_global_normalization_params(self.sub_datasets)
            for ds in self.sub_datasets:
                ds.set_normalization_params(normalization_params)
        self.normalization_params = copy_normalization_params(normalization_params)

        actual_max_points = max(ds.n_points for ds in self.sub_datasets)
        if max_points is not None and max_points < actual_max_points:
            raise ValueError("max_points cannot be smaller than a condition's full point count")
        self.global_max_points = max_points or actual_max_points

        self.n_channels = self.sub_datasets[0].n_channels

        self._build_index_map()

        print(f"\n多工况CFDBench Irregular数据集加载完成 [{split}]:")
        print(f"  工况数量: {self.num_conditions}")
        print(f"  总样本数: {len(self._index_map)}")
        print(f"  全局 max_points: {self.global_max_points}")
        for i, ds in enumerate(self.sub_datasets):
            print(f"  工况 {i} ({ds.benchmark}/{ds.case}): "
                  f"{len(ds)} 样本, {ds.n_points} points")

    def _build_index_map(self):
        self._index_map: List[Tuple[int, int]] = []
        for cond_id, ds in enumerate(self.sub_datasets):
            for local_idx in range(len(ds)):
                self._index_map.append((cond_id, local_idx))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_id, local_idx = self._index_map[idx]
        sub_ds = self.sub_datasets[cond_id]

        pos, fx, y = sub_ds[local_idx]
        return pos, fx, y, torch.tensor(cond_id, dtype=torch.long)

    def get_sub_dataset(self, condition_id: int) -> CFDBenchIrregularDataset:
        return self.sub_datasets[condition_id]

    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        return copy_normalization_params(self.normalization_params)

    def get_global_shape(self) -> Dict[str, int]:
        return {
            'n_channels': self.n_channels,
            'max_points': self.global_max_points,
        }

    @classmethod
    def from_existing(
        cls,
        existing: 'MultiConditionCFDBenchIrregularDataset',
        split: str,
        max_points: Optional[int] = None,
    ) -> 'MultiConditionCFDBenchIrregularDataset':
        new_ds = cls.__new__(cls)
        new_ds.num_conditions = existing.num_conditions
        new_ds.sub_datasets = [
            _clone_dataset_with_split(ds, split) for ds in existing.sub_datasets
        ]

        if max_points is not None:
            new_ds.global_max_points = max_points
        else:
            new_ds.global_max_points = existing.global_max_points

        new_ds.n_channels = new_ds.sub_datasets[0].n_channels
        new_ds.normalization_params = copy_normalization_params(
            existing.normalization_params
        )
        new_ds._build_index_map()

        print(f"\n多工况CFDBench Irregular数据集复用完成 [{split}]:")
        print(f"  工况数量: {new_ds.num_conditions}")
        print(f"  总样本数: {len(new_ds._index_map)}")
        print(f"  全局 max_points: {new_ds.global_max_points}")
        for i, ds in enumerate(new_ds.sub_datasets):
            print(f"  工况 {i} ({ds.benchmark}/{ds.case}): "
                  f"{len(ds)} 样本, {ds.n_points} points")

        return new_ds


class MultiConditionCFDBenchPatchDataset(Dataset):
    """
    多工况CFDBench Patch流场数据集

    支持同时加载多个CFD工况目录的数据:
    1. 不同工况的 num_patches / max_points 不同 -> padding到全局最大值
    2. 不同工况的时间步混合后索引混乱 -> 通过 (condition_id, timestep_id) 双索引

    输出格式与CFDBenchPatchDataset一致,额外返回condition_id

    Args:
        split: 'train' 或 'test'，控制每个子数据集使用哪个划分
    """

    def __init__(
        self,
        roots: List[str],
        benchmarks: List[str],
        cases: List[str],
        step_size: int = 1,
        patch_size: int = 64,
        enable_downsample: bool = True,
        downsample_method: str = 'uniform',
        downsample_ratio: float = 0.25,
        include_coordinates: bool = True,
        output_dim: int = 2,
        normalize: bool = True,
        split: str = 'train',
        train_ratio: float = 0.8,
        seed: int = 42,
        max_patches: Optional[int] = None,
        max_points: Optional[int] = None,
        max_full_points: Optional[int] = None,
        rollout_holdout_steps: int = 0,
        normalization_params: Optional[Dict[str, np.ndarray]] = None,
    ):
        assert len(roots) == len(benchmarks) == len(cases), \
            "roots, benchmarks, cases must have the same length"

        self.num_conditions = len(roots)

        shared_kwargs = dict(
            step_size=step_size,
            patch_size=patch_size,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
            include_coordinates=include_coordinates,
            output_dim=output_dim,
            normalize=normalize,
            rollout_holdout_steps=rollout_holdout_steps,
            normalization_params=normalization_params,
        )

        self.sub_datasets: List[CFDBenchPatchDataset] = []
        for i, (root, benchmark, case) in enumerate(zip(roots, benchmarks, cases)):
            print(f"加载工况 {i} [{split}]: {benchmark}/{case}")
            ds = CFDBenchPatchDataset(
                root=root, benchmark=benchmark, case=case,
                split=split, train_ratio=train_ratio, seed=seed, **shared_kwargs
            )
            self.sub_datasets.append(ds)

        if normalize and normalization_params is None:
            normalization_params = compute_global_normalization_params(self.sub_datasets)
            for ds in self.sub_datasets:
                ds.set_normalization_params(normalization_params)
        self.normalization_params = copy_normalization_params(normalization_params)

        if max_patches is not None:
            self.global_max_patches = max_patches
        else:
            self.global_max_patches = max(ds.num_patches for ds in self.sub_datasets)

        if max_points is not None:
            self.global_max_points = max_points
        else:
            self.global_max_points = max(ds.max_points for ds in self.sub_datasets)

        if max_full_points is not None:
            self.global_max_full_points = max_full_points
        else:
            self.global_max_full_points = max(ds.num_points for ds in self.sub_datasets)

        self.input_dim = self.sub_datasets[0].input_dim
        self.output_dim = self.sub_datasets[0].output_dim

        self._build_index_map()

        print(f"\n多工况CFDBench Patch数据集加载完成 [{split}]:")
        print(f"  工况数量: {self.num_conditions}")
        print(f"  总样本数: {len(self._index_map)}")
        print(f"  全局 max_patches: {self.global_max_patches}")
        print(f"  全局 max_points: {self.global_max_points}")
        for i, ds in enumerate(self.sub_datasets):
            print(f"  工况 {i} ({ds.benchmark}/{ds.case}): "
                  f"{len(ds)} 样本, {ds.num_patches} patches, {ds.max_points} max_points")

    def _build_index_map(self):
        self._index_map: List[Tuple[int, int]] = []
        for cond_id, ds in enumerate(self.sub_datasets):
            for local_idx in range(len(ds)):
                self._index_map.append((cond_id, local_idx))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cond_id, local_idx = self._index_map[idx]
        sub_ds = self.sub_datasets[cond_id]

        sample = sub_ds[local_idx]

        orig_input = sample['input']
        orig_output = sample['output']
        orig_mask = sample['mask']

        input_padded = torch.zeros(
            (self.global_max_patches, self.global_max_points * self.input_dim),
            dtype=torch.float32
        )
        output_padded = torch.zeros(
            (self.global_max_patches, self.global_max_points * self.output_dim),
            dtype=torch.float32
        )
        mask_padded = torch.zeros(
            (self.global_max_patches, self.global_max_points),
            dtype=torch.bool
        )
        full_target_padded = torch.zeros(
            (self.global_max_full_points, self.output_dim), dtype=torch.float32
        )
        full_point_mask = torch.zeros(self.global_max_full_points, dtype=torch.bool)

        p = orig_input.shape[0]
        n = orig_mask.shape[1]
        p_use = min(p, self.global_max_patches)
        n_use = min(n, self.global_max_points)

        input_padded[:p_use, :n_use * self.input_dim] = orig_input[:p_use, :n_use * self.input_dim]
        output_padded[:p_use, :n_use * self.output_dim] = orig_output[:p_use, :n_use * self.output_dim]
        mask_padded[:p_use, :n_use] = orig_mask[:p_use, :n_use]
        n_full = min(sample['full_target'].shape[0], self.global_max_full_points)
        full_target_padded[:n_full] = sample['full_target'][:n_full]
        full_point_mask[:n_full] = True

        return {
            'input': input_padded,
            'output': output_padded,
            'mask': mask_padded,
            'full_target': full_target_padded,
            'full_point_mask': full_point_mask,
            'condition_id': torch.tensor(cond_id, dtype=torch.long),
            'time_index': sample['time_index'],
        }

    def get_sub_dataset(self, condition_id: int) -> CFDBenchPatchDataset:
        return self.sub_datasets[condition_id]

    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        return copy_normalization_params(self.normalization_params)

    def get_global_shape(self) -> Dict[str, int]:
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_patches': self.global_max_patches,
            'max_points': self.global_max_points,
            'full_points': self.global_max_full_points,
        }

    @classmethod
    def from_existing(
        cls,
        existing: 'MultiConditionCFDBenchPatchDataset',
        split: str,
        max_patches: Optional[int] = None,
        max_points: Optional[int] = None,
    ) -> 'MultiConditionCFDBenchPatchDataset':
        new_ds = cls.__new__(cls)
        new_ds.num_conditions = existing.num_conditions
        new_ds.sub_datasets = [
            _clone_dataset_with_split(ds, split) for ds in existing.sub_datasets
        ]

        if max_patches is not None:
            new_ds.global_max_patches = max_patches
        else:
            new_ds.global_max_patches = existing.global_max_patches

        if max_points is not None:
            new_ds.global_max_points = max_points
        else:
            new_ds.global_max_points = existing.global_max_points

        new_ds.global_max_full_points = existing.global_max_full_points
        new_ds.input_dim = new_ds.sub_datasets[0].input_dim
        new_ds.output_dim = new_ds.sub_datasets[0].output_dim
        new_ds.normalization_params = copy_normalization_params(
            existing.normalization_params
        )
        new_ds._build_index_map()

        print(f"\n多工况CFDBench Patch数据集复用完成 [{split}]:")
        print(f"  工况数量: {new_ds.num_conditions}")
        print(f"  总样本数: {len(new_ds._index_map)}")
        print(f"  全局 max_patches: {new_ds.global_max_patches}")
        print(f"  全局 max_points: {new_ds.global_max_points}")
        for i, ds in enumerate(new_ds.sub_datasets):
            print(f"  工况 {i} ({ds.benchmark}/{ds.case}): "
                  f"{len(ds)} 样本, {ds.num_patches} patches, {ds.max_points} max_points")

        return new_ds
