"""
CFDBench Dataset Loader

Supports loading CFD Bench data (03_damflow, 04_cylinderflow, etc.)
with both irregular and patch formats aligned with shipBench interface.

Data format per txt file:
    nodenumber  x-coordinate  y-coordinate  water-vof  y-velocity  x-velocity

Split/normalization/recovery plumbing lives in base.py; the protocol
invariant is that normalization statistics always come from the training
pairs of the temporal split only.
"""

import os

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from torch.utils.data import DataLoader

from ..data_processor.mesh_quad import QuadTreeMesh
from .base import (
    IrregularPairDataset,
    MultiConditionIrregularDatasetMixin,
    MultiConditionPatchDatasetMixin,
    PatchPairDataset,
)
from .temporal import (
    compute_global_normalization_params,
    copy_normalization_params,
    stable_condition_seed,
)


def _load_npz_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if not os.path.exists(cache_path):
        return None

    data = np.load(cache_path, allow_pickle=False)
    if "coords" not in data or "flows" not in data or "channels" not in data:
        raise ValueError(
            f"Incomplete CFDBench cache: {cache_path}; rebuild it with "
            "scripts/build_cfdbench_flow_cache.py"
        )

    coords = data["coords"].astype(np.float32, copy=False)
    flows = data["flows"].astype(np.float32, copy=False)
    channels = [str(c) for c in data["channels"].tolist()]
    return coords, flows, channels


CFD_COLUMN_ALIASES = {
    "volume-fraction-water": "water-vof",
    "y-velocity-water": "y-velocity",
    "x-velocity-water": "x-velocity",
}


def _canonicalize_cfd_channel_name(name: str) -> str:
    return CFD_COLUMN_ALIASES.get(name, name)


def _canonicalize_cfd_channels(names: list[str]) -> list[str]:
    return [_canonicalize_cfd_channel_name(name) for name in names]


def _standardize_cfd_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for alias, canonical in CFD_COLUMN_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            rename_map[alias] = canonical

    if rename_map:
        return df.rename(columns=rename_map)
    return df


class CFDBenchIrregularDataset(IrregularPairDataset):
    """
    CFDBench Irregular format dataset

    Output format:
        pos: (N, 2) - x, y coordinates
        fx: (N, C) - flow features at time t
        y: (N, C) - flow features at time t+1
    """

    def __init__(
        self,
        root: str,
        benchmark: str = "03_damflow",
        case: str = "case0",
        step_size: int = 1,
        train_ratio: float = 0.8,
        seed: int = 42,
        output_channels: list[str] | None = None,
        normalize: bool = True,
        prefer_cache: bool = True,
        cache_filename: str = "flow_cache.npz",
        rollout_holdout_steps: int = 0,
        normalization_params: dict | None = None,
        _defer_normalization: bool = False,
    ):
        self.root = root
        self.benchmark = benchmark
        self.case = case
        self.split = "train"
        self.normalize = normalize
        self.prefer_cache = prefer_cache
        self.cache_filename = cache_filename
        self.output_channels = _canonicalize_cfd_channels(
            output_channels or ["y-velocity", "x-velocity"]
        )
        self._init_temporal_pair(
            step_size=step_size,
            train_ratio=train_ratio,
            split_seed=stable_condition_seed(seed, os.path.join(root, benchmark, case)),
            rollout_holdout_steps=rollout_holdout_steps,
            normalization_params=normalization_params,
            defer_normalization=_defer_normalization,
        )

        self._load_data()
        self._split_data()
        if not self._defer_normalization:
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
                self.n_points = self.coords.shape[1]
                self.n_channels = self.flows.shape[2]
                return

        data_files = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.startswith("data") and f.endswith(".txt")
            ]
        )

        coords_list = []
        flows_list = []

        for data_file in data_files:
            data_path = os.path.join(data_dir, data_file)
            df = pd.read_csv(data_path, sep=r"\s+")
            df = _standardize_cfd_columns(df)

            coords = df[["x-coordinate", "y-coordinate"]].to_numpy(dtype=np.float32)

            flow_cols = [c for c in self.output_channels if c in df.columns]
            if not flow_cols:
                raise ValueError(
                    f"No requested output channels in file: {data_path}; "
                    f"requested: {self.output_channels}; available: {list(df.columns)}"
                )
            flows = df[flow_cols].to_numpy(dtype=np.float32)

            coords_list.append(coords)
            flows_list.append(flows)

        min_cells = min(len(c) for c in coords_list)
        reference_coords = coords_list[0]

        sorted_coords = []
        sorted_flows = []

        for coords, flows in zip(coords_list, flows_list, strict=True):
            tree = KDTree(coords)
            _distance, indices = tree.query(reference_coords[:min_cells])
            sorted_coords.append(coords[indices])
            sorted_flows.append(flows[indices])

        self.coords = np.array(sorted_coords, dtype=np.float32)
        self.flows = np.array(sorted_flows, dtype=np.float32)

        self.n_points = min_cells
        self.n_channels = self.flows.shape[2]


class CFDBenchPatchDataset(PatchPairDataset):
    """
    CFDBench Patch format dataset

    Output format:
        input: (P, N*C) flattened patches
        output: (P, N*C) flattened patches
        mask: (P, N) valid point mask
    """

    def __init__(
        self,
        root: str,
        benchmark: str = "03_damflow",
        case: str = "case0",
        step_size: int = 1,
        patch_size: int = 64,
        enable_downsample: bool = True,
        downsample_method: str = "uniform",
        downsample_ratio: float = 0.25,
        include_coordinates: bool = True,
        output_dim: int = 2,
        normalize: bool = True,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        prefer_cache: bool = True,
        cache_filename: str = "flow_cache.npz",
        rollout_holdout_steps: int = 0,
        normalization_params: dict | None = None,
        _defer_normalization: bool = False,
    ):
        self.root = root
        self.benchmark = benchmark
        self.case = case
        self.split = split
        self.normalize = normalize
        self.patch_size = patch_size
        self.enable_downsample = enable_downsample
        self.downsample_method = downsample_method
        self.downsample_ratio = downsample_ratio
        self.include_coordinates = include_coordinates
        self.output_dim = output_dim
        self.prefer_cache = prefer_cache
        self.cache_filename = cache_filename
        self._init_temporal_pair(
            step_size=step_size,
            train_ratio=train_ratio,
            split_seed=stable_condition_seed(seed, os.path.join(root, benchmark, case)),
            rollout_holdout_steps=rollout_holdout_steps,
            normalization_params=normalization_params,
            defer_normalization=_defer_normalization,
        )

        self._load_data()
        self._split_data()
        self._build_quadtree()
        if not self._defer_normalization:
            self._compute_normalization_params()

    def _load_data(self):
        data_dir = os.path.join(self.root, self.benchmark, self.case)

        if self.prefer_cache:
            cached = _load_npz_cache(os.path.join(data_dir, self.cache_filename))
            if cached is not None:
                coords, all_flows, channel_names = cached
                channel_names = _canonicalize_cfd_channels(channel_names)
                channel_to_idx = {name: i for i, name in enumerate(channel_names)}
                required = ["y-velocity", "x-velocity"]
                if not all(ch in channel_to_idx for ch in required[: self.output_dim]):
                    raise ValueError(
                        f"Cache missing required CFD channels for output_dim={self.output_dim}; "
                        f"available: {channel_names}"
                    )

                sel_idx = [channel_to_idx[ch] for ch in required[: self.output_dim]]
                self.coords = coords
                self.flows = all_flows[:, :, sel_idx]
                self.num_points = self.coords.shape[1]
                return

        data_files = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.startswith("data") and f.endswith(".txt")
            ]
        )

        coords_list = []
        flows_list = []

        for data_file in data_files:
            data_path = os.path.join(data_dir, data_file)
            df = pd.read_csv(data_path, sep=r"\s+")
            df = _standardize_cfd_columns(df)

            coords = df[["x-coordinate", "y-coordinate"]].to_numpy(dtype=np.float32)

            flow_values = df[["y-velocity", "x-velocity"]].to_numpy(dtype=np.float32)
            flow_values = flow_values[:, : self.output_dim]

            coords_list.append(coords)
            flows_list.append(flow_values)

        min_cells = min(len(c) for c in coords_list)
        reference_coords = coords_list[0]

        sorted_coords = []
        sorted_flows = []

        for coords, flows in zip(coords_list, flows_list, strict=True):
            tree = KDTree(coords)
            _distance, indices = tree.query(reference_coords[:min_cells])
            sorted_coords.append(coords[indices])
            sorted_flows.append(flows[indices])

        self.coords = np.array(sorted_coords, dtype=np.float32)
        self.flows = np.array(sorted_flows, dtype=np.float32)

        self.num_points = min_cells

    def _build_quadtree(self):
        self.quadtree = QuadTreeMesh(
            self.coords[0],
            patch_size=self.patch_size,
            enable_distance_refine=False,
        )

        if self.enable_downsample:
            target_points = max(4, int(self.patch_size * self.downsample_ratio))
            self.quadtree.downsample_patches_by_distance(
                method=self.downsample_method, target_points=target_points, min_points=4
            )

        self.num_patches = len(self.quadtree.patches)
        self.max_points = max(len(p.points) for p in self.quadtree.patches)
        self._build_recovery_map()
        self._build_patch_index()

        self.coord_dim = 2 if self.include_coordinates else 0
        self.input_dim = self.coord_dim + self.output_dim

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        time_idx = int(self.pair_indices[idx])
        input_flows = self._all_flows[time_idx]
        output_flows = self._all_flows[time_idx + self.step_size]

        coords = self._all_coords[time_idx]
        coords_norm = self._normalize_coords(coords)
        input_flows_norm = self._normalize_flows(input_flows)
        output_flows_norm = self._normalize_flows(output_flows)

        return {
            "input": self._extract_patches(coords_norm, input_flows_norm),
            "output": self._extract_flow_patches(output_flows_norm),
            "mask": self._create_mask(),
            "full_target": torch.from_numpy(output_flows_norm).float(),
            "time_index": torch.tensor(time_idx, dtype=torch.long),
        }


def create_cfd_bench_irregular_dataloader(
    root: str,
    benchmark: str = "03_damflow",
    case: str = "case0",
    split: str = "train",
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
    benchmark: str = "03_damflow",
    case: str = "case0",
    split: str = "train",
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


class MultiConditionCFDBenchIrregularDataset(MultiConditionIrregularDatasetMixin):
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
        roots: list[str],
        benchmarks: list[str],
        cases: list[str],
        step_size: int = 1,
        train_ratio: float = 0.8,
        seed: int = 42,
        output_channels: list[str] | None = None,
        normalize: bool = True,
        split: str = "train",
        max_points: int | None = None,
        rollout_holdout_steps: int = 0,
        normalization_params: dict | None = None,
    ):
        assert len(roots) == len(benchmarks) == len(cases), (
            "roots, benchmarks, cases must have the same length"
        )

        self.num_conditions = len(roots)

        # The wrapper computes global statistics itself when none are given.
        defer_normalization = normalize and normalization_params is None

        self.sub_datasets: list[CFDBenchIrregularDataset] = []
        for i, (root, benchmark, case) in enumerate(
            zip(roots, benchmarks, cases, strict=True)
        ):
            print(f"加载工况 {i} [{split}]: {benchmark}/{case}")
            ds = CFDBenchIrregularDataset(
                root=root,
                benchmark=benchmark,
                case=case,
                seed=seed,
                step_size=step_size,
                train_ratio=train_ratio,
                output_channels=output_channels,
                normalize=normalize,
                rollout_holdout_steps=rollout_holdout_steps,
                normalization_params=normalization_params,
                _defer_normalization=defer_normalization,
            )
            ds.set_split(split)
            self.sub_datasets.append(ds)

        if normalize and normalization_params is None:
            normalization_params = compute_global_normalization_params(
                self.sub_datasets
            )
            for ds in self.sub_datasets:
                ds.set_normalization_params(normalization_params)
        self.normalization_params = copy_normalization_params(normalization_params)

        actual_max_points = max(ds.n_points for ds in self.sub_datasets)
        if max_points is not None and max_points < actual_max_points:
            raise ValueError(
                "max_points cannot be smaller than a condition's full point count"
            )
        self.global_max_points = max_points or actual_max_points

        self.n_channels = self.sub_datasets[0].n_channels

        self._build_index_map()

        print(f"\n多工况CFDBench Irregular数据集加载完成 [{split}]:")
        print(f"  工况数量: {self.num_conditions}")
        print(f"  总样本数: {len(self._index_map)}")
        print(f"  全局 max_points: {self.global_max_points}")
        for i, ds in enumerate(self.sub_datasets):
            print(
                f"  工况 {i} ({ds.benchmark}/{ds.case}): "
                f"{len(ds)} 样本, {ds.n_points} points"
            )

    @classmethod
    def from_existing(cls, source, split: str, max_points: int | None = None):
        dataset = super().from_existing(source, split)
        if max_points is not None:
            dataset.global_max_points = max_points
        return dataset


class MultiConditionCFDBenchPatchDataset(MultiConditionPatchDatasetMixin):
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
        roots: list[str],
        benchmarks: list[str],
        cases: list[str],
        step_size: int = 1,
        patch_size: int = 64,
        enable_downsample: bool = True,
        downsample_method: str = "uniform",
        downsample_ratio: float = 0.25,
        include_coordinates: bool = True,
        output_dim: int = 2,
        normalize: bool = True,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        max_patches: int | None = None,
        max_points: int | None = None,
        max_full_points: int | None = None,
        rollout_holdout_steps: int = 0,
        normalization_params: dict | None = None,
    ):
        assert len(roots) == len(benchmarks) == len(cases), (
            "roots, benchmarks, cases must have the same length"
        )

        self.num_conditions = len(roots)

        # The wrapper computes global statistics itself when none are given.
        defer_normalization = normalize and normalization_params is None

        self.sub_datasets: list[CFDBenchPatchDataset] = []
        for i, (root, benchmark, case) in enumerate(
            zip(roots, benchmarks, cases, strict=True)
        ):
            print(f"加载工况 {i} [{split}]: {benchmark}/{case}")
            ds = CFDBenchPatchDataset(
                root=root,
                benchmark=benchmark,
                case=case,
                split=split,
                seed=seed,
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
                _defer_normalization=defer_normalization,
            )
            self.sub_datasets.append(ds)

        if normalize and normalization_params is None:
            normalization_params = compute_global_normalization_params(
                self.sub_datasets
            )
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
            print(
                f"  工况 {i} ({ds.benchmark}/{ds.case}): "
                f"{len(ds)} 样本, {ds.num_patches} patches, {ds.max_points} max_points"
            )

    @classmethod
    def from_existing(
        cls,
        source,
        split: str,
        max_patches: int | None = None,
        max_points: int | None = None,
    ):
        dataset = super().from_existing(source, split)
        if max_patches is not None:
            dataset.global_max_patches = max_patches
        if max_points is not None:
            dataset.global_max_points = max_points
        return dataset
