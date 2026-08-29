"""
船型流场数据集

支持两种数据格式：
1. Irregular格式：IrregularFlowFieldDataset - (B, N, C)
2. Patch格式：PatchFlowFieldDataset - (B, C, P, N)

Split/normalization/recovery plumbing lives in base.py; the protocol
invariant is that normalization statistics always come from the training
pairs of the temporal split only.
"""

import os
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
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

_TIMESTEP_RE = re.compile(r"^timestep_(\d+)\.csv$")

DEFAULT_SHIP_CHANNELS = ["U:0", "U:1", "U:2", "p_rgh"]
SHIP_COORD_CHANNELS = ["Center:0", "Center:1", "Center:2"]


def _timestep_sort_key(path: str) -> int:
    match = _TIMESTEP_RE.match(os.path.basename(path))
    if match is None:
        raise ValueError(f"Unexpected timestep filename: {path}")
    return int(match.group(1))


def _align_to_fixed_reference(
    coords_3d: list[np.ndarray], flows: list[np.ndarray], k: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    reference = coords_3d[0]
    aligned_flows = np.empty(
        (len(flows), len(reference), flows[0].shape[1]), dtype=np.float32
    )
    aligned_flows[0] = flows[0]
    for frame_id, (source_coords, source_flows) in enumerate(
        zip(coords_3d[1:], flows[1:], strict=True), start=1
    ):
        # pi-lens-ignore: python-sql-injection
        distances, neighbors = KDTree(source_coords).query(reference, k=k)
        neighbors = np.asarray(neighbors)
        exact = distances[:, 0] <= 1e-12
        aligned_flows[frame_id, exact] = source_flows[neighbors[exact, 0]]
        weights = 1.0 / np.maximum(distances[~exact], 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        aligned_flows[frame_id, ~exact] = np.sum(
            source_flows[neighbors[~exact]] * weights[..., None], axis=1
        )
    fixed_coords = np.broadcast_to(
        reference[None, :, :2], (len(coords_3d), len(reference), 2)
    ).copy()
    return fixed_coords, aligned_flows


def _load_npz_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if not os.path.exists(cache_path):
        return None

    data = np.load(cache_path, allow_pickle=False)
    if "coords" not in data or "flows" not in data or "channels" not in data:
        raise ValueError(f"Incomplete ShipBench cache: {cache_path}")
    if "frame_indices" not in data:
        raise ValueError(
            f"Legacy ShipBench cache without numeric frame indices: {cache_path}; "
            "rebuild it with scripts/rebuild_ship_flow_cache.py"
        )

    coords = data["coords"].astype(np.float32, copy=False)
    flows = data["flows"].astype(np.float32, copy=False)
    frame_indices = data["frame_indices"]
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError(f"Non-contiguous ShipBench cache time order: {cache_path}")
    if not np.array_equal(coords[0], coords[-1]):
        raise ValueError(f"ShipBench cache coordinates are not fixed: {cache_path}")
    channels = [str(c) for c in data["channels"].tolist()]
    return coords, flows, channels


def yaml_to_text(yaml_data: dict, parent_key: str = "") -> str:
    """
    Convert YAML dictionary to natural language text description for LLM.

    Args:
        yaml_data: Parsed YAML dictionary
        parent_key: Parent key for nested dicts

    Returns:
        Natural language text description of the parameters
    """
    lines = []

    param_mappings = {
        "B": "beam",
        "Lpp": "length between perpendiculars",
        "Cb": "block coefficient",
        "Cm": "prismatic coefficient",
        "Cp": "pitch coefficient",
        "Cw": "waterplane coefficient",
        "Loa": "length overall",
        "Lwl": "waterline length",
        "TM": "mean draft",
        "Vol": "volume",
        "Srea": "wetted surface area",
        "KM": "metacentric height",
        "B_bulb": "bulb beam",
        "Lb": "bulb length",
        "ZFPU": "bulb forward upper position",
        "ZFPd": "bulb forward lower position",
        "Zb": "bulb vertical position",
        "dboss": "bulb diameter",
        "vbulb": "bulb volume",
        "xatb": "bulb tip position",
        "xboss": "bulb boss position",
        "xtran": "bulb transition position",
        "xclear": "bulb clearance",
        "B_T": "draft to depth ratio",
        "Cba": "aft block coefficient",
        "Cbf": "fore block coefficient",
        "Cpa": "aft prismatic coefficient",
        "Cpf": "fore prismatic coefficient",
        "Cpv": "vertical prismatic coefficient",
        "S_Lpp": "slenderness ratio",
        "S_Vol": "volume coefficient",
        "Xb": "longitudinal center of buoyancy",
        "g": "gravity",
        "hRef": "reference height",
        "nu": "kinematic viscosity",
        "rho": "density",
        "Umean": "mean velocity",
        "sigma": "surface tension",
    }

    def format_value(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    for key, value in yaml_data.items():
        if isinstance(value, dict):
            section_name = key.replace("_", " ").title()
            lines.append(f"The {section_name} section:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    sub_name = sub_key.replace("_", " ").title()
                    for k, v in sub_value.items():
                        param_name = param_mappings.get(k, k.replace("_", " "))
                        lines.append(f"  {sub_name} {param_name}: {format_value(v)}")
                else:
                    param_name = param_mappings.get(sub_key, sub_key.replace("_", " "))
                    lines.append(f"  {param_name}: {format_value(sub_value)}")
        else:
            param_name = param_mappings.get(key, key.replace("_", " "))
            lines.append(f"{param_name} is {format_value(value)}.")

    return " ".join(lines)


def yaml_to_numeric(yaml_data: dict) -> tuple[list[str], np.ndarray]:
    values = []

    def visit(node, prefix=""):
        for key in sorted(node):
            path = f"{prefix}.{key}" if prefix else key
            value = node[key]
            if isinstance(value, dict):
                visit(value, path)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append((path, float(value)))

    visit(yaml_data)
    keys = [key for key, _value in values]
    vector = np.asarray([value for _key, value in values], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Ship condition parameters contain non-finite values")
    return keys, vector


def find_ship_params_file(data_dir: str, params_path: str | None = None) -> str:
    if params_path is not None:
        if not os.path.isfile(params_path):
            raise FileNotFoundError(params_path)
        return params_path
    matches = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.startswith("ship_params_") and name.endswith(".yaml")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ship_params_*.yaml in {data_dir}, found {len(matches)}"
        )
    return matches[0]


def copy_condition_normalization_params(params):
    """Copy condition-stat dicts so callers cannot mutate dataset state."""
    if params is None:
        return None
    return {
        "keys": np.asarray(params["keys"]).copy(),
        "mean": np.asarray(params["mean"], dtype=np.float32).copy(),
        "std": np.asarray(params["std"], dtype=np.float32).copy(),
    }


class IrregularFlowFieldDataset(IrregularPairDataset):
    """
    Transolver格式的流场数据集 - Irregular输入

    输出格式:
        pos: (N, 2) 归一化坐标
        fx: (N, C) 归一化流场特征
        y: (N, C) 归一化目标流场
    """

    def __init__(
        self,
        data_dir: str,
        step_size: int = 1,
        train_ratio: float = 0.8,
        seed: int = 42,
        output_channels: list[str] | None = None,
        normalize: bool = True,
        prefer_cache: bool = True,
        cache_filename: str = "flow_cache.npz",
        rollout_holdout_steps: int = 0,
        normalization_params: dict[str, np.ndarray] | None = None,
        _defer_normalization: bool = False,
    ):
        self.split = "train"
        self.normalize = normalize
        self.prefer_cache = prefer_cache
        self.cache_filename = cache_filename
        self.output_channels = output_channels or list(DEFAULT_SHIP_CHANNELS)
        self.coords_channels = list(SHIP_COORD_CHANNELS)
        self._init_temporal_pair(
            step_size=step_size,
            train_ratio=train_ratio,
            split_seed=stable_condition_seed(seed, data_dir),
            rollout_holdout_steps=rollout_holdout_steps,
            normalization_params=normalization_params,
            defer_normalization=_defer_normalization,
        )

        self._load_from_csv(data_dir)
        self._split_data()
        if not self._defer_normalization:
            self._compute_normalization_params()

    def _load_from_csv(self, data_dir: str):
        if self.prefer_cache:
            cached = _load_npz_cache(os.path.join(data_dir, self.cache_filename))
            if cached is not None:
                coords, all_flows, channel_names = cached
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
                self.data_dir = data_dir
                self.n_points = self.coords.shape[1]
                self.n_channels = self.flows.shape[2]
                return

        timestep_files = sorted(
            (
                f
                for f in os.listdir(data_dir)
                if f.startswith("timestep_") and f.endswith(".csv")
            ),
            key=_timestep_sort_key,
        )

        coords_list = []
        flows_list = []

        for ts_file in timestep_files:
            ts_path = os.path.join(data_dir, ts_file)
            df = pd.read_csv(ts_path)

            coords = df[self.coords_channels].to_numpy(dtype=np.float32)

            flow_cols = [c for c in self.output_channels if c in df.columns]
            flows = df[flow_cols].to_numpy(dtype=np.float32)

            coords_list.append(coords)
            flows_list.append(flows)

        self.coords, self.flows = _align_to_fixed_reference(coords_list, flows_list)

        self.data_dir = data_dir
        self.n_points = self.coords.shape[1]
        self.n_channels = self.flows.shape[2]


class PatchFlowFieldDataset(PatchPairDataset):
    """
    Patch格式的流场数据集 - Patch输入

    输出格式:
        input: (P, N*C_in) 输入patch，包含坐标和流场
        output: (P, N*C_out) 目标patch，仅包含下一步流场
        mask: (P, N) 有效点mask
        params_text: str 船舶参数字本描述 (当enable_params=True时)
    """

    def __init__(
        self,
        data_dir: str,
        step_size: int = 1,
        patch_size: int = 64,
        ship_length: float = 7.0,
        ref_point: tuple[float, float] = (3.0, 0.0),
        distance_threshold_1: float = 1.0,
        distance_threshold_2: float = 1.5,
        enable_distance_refine: bool = True,
        enable_downsample: bool = True,
        downsample_method: str = "uniform",
        downsample_ratio: float = 0.25,
        include_coordinates: bool = True,
        output_dim: int = 4,
        normalize: bool = True,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        enable_params: bool = False,
        params_path: str | None = None,
        embedding_filename: str = "ship_params_embedding.pt",
        embedding_mode: str = "precomputed",
        zero_embedding_dim: int = 0,
        prefer_cache: bool = True,
        cache_filename: str = "flow_cache.npz",
        rollout_holdout_steps: int = 0,
        normalization_params: dict[str, np.ndarray] | None = None,
        condition_normalization_params: dict[str, np.ndarray] | None = None,
        _defer_normalization: bool = False,
    ):
        self.data_dir = data_dir
        self.split = split
        self.normalize = normalize
        self.patch_size = patch_size
        self.ship_length = ship_length
        self.ref_point = ref_point
        self.distance_threshold_1 = distance_threshold_1
        self.distance_threshold_2 = distance_threshold_2
        self.enable_distance_refine = enable_distance_refine
        self.enable_downsample = enable_downsample
        self.downsample_method = downsample_method
        self.downsample_ratio = downsample_ratio
        self.include_coordinates = include_coordinates
        self.output_dim = output_dim
        self.enable_params = enable_params
        self.params_path = params_path
        self.embedding_filename = embedding_filename
        self.embedding_mode = embedding_mode
        self.zero_embedding_dim = zero_embedding_dim
        self.prefer_cache = prefer_cache
        self.cache_filename = cache_filename
        self._condition_normalization_params = copy_condition_normalization_params(
            condition_normalization_params
        )
        self._init_temporal_pair(
            step_size=step_size,
            train_ratio=train_ratio,
            split_seed=stable_condition_seed(seed, data_dir),
            rollout_holdout_steps=rollout_holdout_steps,
            normalization_params=normalization_params,
            defer_normalization=_defer_normalization,
        )

        self.timestep_files = self._get_timestep_files()

        self._load_and_align_data()
        self._split_data()
        self._build_quadtree()
        if not self._defer_normalization:
            self._compute_normalization_params()

        if self.enable_params:
            self._load_ship_params()

    def _get_timestep_files(self) -> list[str]:
        files = sorted(
            (
                f
                for f in os.listdir(self.data_dir)
                if f.startswith("timestep_") and f.endswith(".csv")
            ),
            key=_timestep_sort_key,
        )
        return [os.path.join(self.data_dir, f) for f in files]

    def _load_timestep(self, timestep_file: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(timestep_file)

        coords = df[list(SHIP_COORD_CHANNELS)].to_numpy(dtype=np.float32)

        requested = DEFAULT_SHIP_CHANNELS[: self.output_dim]
        flow_values = df[requested].to_numpy(dtype=np.float32)

        return coords, flow_values

    def _load_and_align_data(self) -> None:
        if self.prefer_cache:
            cached = _load_npz_cache(os.path.join(self.data_dir, self.cache_filename))
            if cached is not None:
                coords, all_flows, channel_names = cached
                # Select cache channels by name, mirroring the irregular path.
                requested = DEFAULT_SHIP_CHANNELS[: self.output_dim]
                channel_to_idx = {name: i for i, name in enumerate(channel_names)}
                missing = [name for name in requested if name not in channel_to_idx]
                if missing:
                    raise ValueError(
                        f"Requested output channels missing from cache: {missing}; "
                        f"available: {channel_names}"
                    )
                self.coords = coords
                self.flows = all_flows[:, :, [channel_to_idx[c] for c in requested]]
                self.num_points = self.coords.shape[1]
                return

        all_coords = []
        all_flows = []

        for ts_file in self.timestep_files:
            coords, flows = self._load_timestep(ts_file)
            all_coords.append(coords)
            all_flows.append(flows)

        self.coords, self.flows = _align_to_fixed_reference(all_coords, all_flows)

        self.num_points = self.coords.shape[1]

    def _build_quadtree(self) -> None:
        self.quadtree = QuadTreeMesh(
            self.coords[0],
            patch_size=self.patch_size,
            ship_length=self.ship_length,
            ref_point=self.ref_point,
            distance_threshold_1=self.distance_threshold_1,
            distance_threshold_2=self.distance_threshold_2,
            enable_distance_refine=self.enable_distance_refine,
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

    def _load_ship_params(self) -> None:
        if self.embedding_mode == "zero":
            if self.zero_embedding_dim <= 0:
                raise ValueError(
                    "zero_embedding_dim must be > 0 when embedding_mode='zero'"
                )
            self.params_embedding = torch.zeros(
                self.zero_embedding_dim, dtype=torch.float32
            )
            self.params_text = "Zero parameter embedding."
            self.embedding_dim = self.zero_embedding_dim
            return

        if self.embedding_mode == "precomputed":
            embedding_path = os.path.join(self.data_dir, self.embedding_filename)
            if not os.path.isfile(embedding_path):
                raise FileNotFoundError(embedding_path)
            data = torch.load(embedding_path, map_location="cpu", weights_only=True)
            self.params_embedding = data["embedding"].float()
            self.params_text = data.get("params_text", "")
            self.embedding_dim = self.params_embedding.shape[-1]
            print(
                f"Loaded pre-computed embedding from {embedding_path}, "
                f"shape: {self.params_embedding.shape}"
            )
            return

        if self.embedding_mode == "numeric":
            yaml_path = find_ship_params_file(self.data_dir, self.params_path)
            with open(yaml_path, encoding="utf-8") as handle:
                yaml_data = yaml.safe_load(handle)
            self.condition_keys, self.raw_params_vector = yaml_to_numeric(yaml_data)
            self.params_text = yaml_to_text(yaml_data)
            self.embedding_dim = len(self.raw_params_vector)
            if self._condition_normalization_params is not None:
                self.set_condition_normalization_params(
                    self._condition_normalization_params
                )
            else:
                self.params_embedding = torch.from_numpy(self.raw_params_vector.copy())
            return

        raise ValueError(f"Unknown embedding_mode: {self.embedding_mode}")

    def set_condition_normalization_params(self, params: dict[str, np.ndarray]) -> None:
        if self.embedding_mode != "numeric":
            return
        keys = list(params["keys"])
        if keys != self.condition_keys:
            raise ValueError("Numeric ship parameter schemas do not match")
        self._condition_normalization_params = {
            "keys": np.asarray(keys),
            "mean": np.asarray(params["mean"], dtype=np.float32),
            "std": np.asarray(params["std"], dtype=np.float32),
        }
        normalized = (
            self.raw_params_vector - self._condition_normalization_params["mean"]
        ) / self._condition_normalization_params["std"]
        self.params_embedding = torch.from_numpy(normalized.astype(np.float32))

    def get_condition_normalization_params(self):
        return copy_condition_normalization_params(self._condition_normalization_params)

    def get_params_embedding(self) -> torch.Tensor | None:
        return getattr(self, "params_embedding", None)

    def get_params_text(self) -> str:
        """Return the ship parameters as text description."""
        return getattr(self, "params_text", "")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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

        result: dict[str, Any] = {
            "input": input_patches,
            "output": output_patches,
            "mask": mask,
            "time_index": torch.tensor(time_idx, dtype=torch.long),
        }
        if self.split != "train":
            result["full_target"] = torch.from_numpy(output_flows_norm).float()

        if self.enable_params:
            embedding = self.get_params_embedding()
            if embedding is not None:
                result["params_embedding"] = embedding
            else:
                result["params_text"] = self.get_params_text()

        return result


def create_irregular_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    step_size: int = 1,
    train_ratio: float = 0.8,
    seed: int = 42,
    output_channels: list[str] | None = None,
) -> DataLoader:
    dataset = IrregularFlowFieldDataset(
        data_dir=data_dir,
        step_size=step_size,
        train_ratio=train_ratio,
        seed=seed,
        output_channels=output_channels,
    )
    dataset.set_split(split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_patch_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    step_size: int = 1,
    patch_size: int = 64,
    output_dim: int = 4,
) -> DataLoader:
    dataset = PatchFlowFieldDataset(
        data_dir=data_dir,
        split=split,
        step_size=step_size,
        patch_size=patch_size,
        output_dim=output_dim,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


class MultiConditionIrregularDataset(MultiConditionIrregularDatasetMixin):
    """
    多工况Irregular流场数据集

    支持同时加载多个工况目录的数据:
    1. 保留每个工况的完整点集，由 ConditionBatchSampler 按工况组 batch
    2. 不同工况的时间样本通过 (condition_id, local_id) 双索引组织

    输出格式与IrregularFlowFieldDataset一致:
        pos: (N, 2) 归一化坐标
        fx: (N, C) 归一化流场特征
        y: (N, C) 归一化目标流场
        condition_id: int 工况编号

    Args:
        data_dirs: 工况目录列表
        split: 'train' 或 'test'，控制每个子数据集使用哪个划分
        max_points: 仅用于记录全局最大点数；数据不会截断
    """

    def __init__(
        self,
        data_dirs: list[str],
        step_size: int = 1,
        train_ratio: float = 0.8,
        seed: int = 42,
        output_channels: list[str] | None = None,
        normalize: bool = True,
        split: str = "train",
        max_points: int | None = None,
        rollout_holdout_steps: int = 0,
        normalization_params: dict[str, np.ndarray] | None = None,
    ):
        self.data_dirs = data_dirs
        self.num_conditions = len(data_dirs)

        # The wrapper computes global statistics itself when none are given.
        defer_normalization = normalize and normalization_params is None

        self.sub_datasets: list[IrregularFlowFieldDataset] = []
        for i, data_dir in enumerate(data_dirs):
            print(f"加载工况 {i} [{split}]: {data_dir}")
            ds = IrregularFlowFieldDataset(
                data_dir=data_dir,
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

        print(f"\n多工况Irregular数据集加载完成 [{split}]:")
        print(f"  工况数量: {self.num_conditions}")
        print(f"  总样本数: {len(self._index_map)}")
        print(f"  全局 max_points: {self.global_max_points}")
        for i, ds in enumerate(self.sub_datasets):
            print(
                f"  工况 {i} ({os.path.basename(ds.data_dir)}): "
                f"{len(ds)} 样本, {ds.n_points} points"
            )


class MultiConditionPatchDataset(MultiConditionPatchDatasetMixin):
    """
    多工况Patch流场数据集

    支持同时加载多个工况目录的数据:
    1. 不同工况的 num_patches / max_points 不同 -> padding到全局最大值
    2. 不同工况的时间步混合后索引混乱 -> 通过 (condition_id, timestep_id) 双索引

    输出格式与PatchFlowFieldDataset一致,额外返回condition_id

    Args:
        split: 'train' 或 'test'，控制每个子数据集使用哪个划分
        max_patches: 全局最大patch数，None时取各工况最大值
        max_points:  全局最大点数，None时取各工况最大值
    """

    def __init__(
        self,
        data_dirs: list[str],
        step_size: int = 1,
        patch_size: int = 64,
        ship_length: float = 7.0,
        ref_point: tuple[float, float] = (3.0, 0.0),
        distance_threshold_1: float = 1.0,
        distance_threshold_2: float = 1.5,
        enable_distance_refine: bool = True,
        enable_downsample: bool = True,
        downsample_method: str = "uniform",
        downsample_ratio: float = 0.25,
        include_coordinates: bool = True,
        output_dim: int = 4,
        normalize: bool = True,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        enable_params: bool = False,
        params_path: str | None = None,
        embedding_filename: str = "ship_params_embedding.pt",
        embedding_mode: str = "precomputed",
        zero_embedding_dim: int = 0,
        prefer_cache: bool = True,
        cache_filename: str = "flow_cache.npz",
        max_patches: int | None = None,
        max_points: int | None = None,
        max_full_points: int | None = None,
        rollout_holdout_steps: int = 0,
        normalization_params: dict[str, np.ndarray] | None = None,
        condition_normalization_params: dict[str, np.ndarray] | None = None,
    ):
        self.data_dirs = data_dirs
        self.num_conditions = len(data_dirs)

        # The wrapper computes global statistics itself when none are given.
        defer_normalization = normalize and normalization_params is None

        self.sub_datasets: list[PatchFlowFieldDataset] = []
        for i, data_dir in enumerate(data_dirs):
            print(f"加载工况 {i} [{split}]: {data_dir}")
            ds = PatchFlowFieldDataset(
                data_dir=data_dir,
                split=split,
                seed=seed,
                step_size=step_size,
                patch_size=patch_size,
                ship_length=ship_length,
                ref_point=ref_point,
                distance_threshold_1=distance_threshold_1,
                distance_threshold_2=distance_threshold_2,
                enable_distance_refine=enable_distance_refine,
                enable_downsample=enable_downsample,
                downsample_method=downsample_method,
                downsample_ratio=downsample_ratio,
                include_coordinates=include_coordinates,
                output_dim=output_dim,
                normalize=normalize,
                enable_params=enable_params,
                params_path=params_path,
                embedding_filename=embedding_filename,
                embedding_mode=embedding_mode,
                zero_embedding_dim=zero_embedding_dim,
                prefer_cache=prefer_cache,
                cache_filename=cache_filename,
                rollout_holdout_steps=rollout_holdout_steps,
                normalization_params=normalization_params,
                condition_normalization_params=condition_normalization_params,
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
        self.condition_normalization_params = None
        if enable_params and embedding_mode == "numeric":
            if condition_normalization_params is None:
                schema = self.sub_datasets[0].condition_keys
                if any(ds.condition_keys != schema for ds in self.sub_datasets):
                    raise ValueError("Numeric ship parameter schemas do not match")
                values = np.stack(
                    [ds.raw_params_vector for ds in self.sub_datasets], axis=0
                )
                mean = values.mean(axis=0)
                std = values.std(axis=0)
                std[std < 1e-8] = 1.0
                condition_normalization_params = {
                    "keys": np.asarray(schema),
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                }
            for ds in self.sub_datasets:
                ds.set_condition_normalization_params(condition_normalization_params)
            self.condition_normalization_params = copy_condition_normalization_params(
                condition_normalization_params
            )

        self.embedding_dim = 0
        if enable_params:
            embedding_dims = [
                getattr(ds, "embedding_dim", 0) for ds in self.sub_datasets
            ]
            nonzero_dims = [dim for dim in embedding_dims if dim > 0]
            if nonzero_dims:
                if len(nonzero_dims) != len(embedding_dims):
                    raise ValueError(
                        f"Some condition directories are missing embeddings: {embedding_dims}"
                    )
                if len(set(nonzero_dims)) != 1:
                    raise ValueError(
                        f"Inconsistent ship parameter embedding dims: {embedding_dims}"
                    )
                self.embedding_dim = nonzero_dims[0]

        self._build_index_map()

        print(f"\n多工况Patch数据集加载完成 [{split}]:")
        print(f"  工况数量: {self.num_conditions}")
        print(f"  总样本数: {len(self._index_map)}")
        print(f"  全局 max_patches: {self.global_max_patches}")
        print(f"  全局 max_points: {self.global_max_points}")
        for i, ds in enumerate(self.sub_datasets):
            print(
                f"  工况 {i} ({os.path.basename(ds.data_dir)}): "
                f"{len(ds)} 样本, {ds.num_patches} patches, {ds.max_points} max_points"
            )

    def get_condition_normalization_params(self):
        return copy_condition_normalization_params(self.condition_normalization_params)
