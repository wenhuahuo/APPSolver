"""Shared temporal-pair dataset plumbing for the ShipBench/CFDBench loaders.

The protocol invariant implemented here: normalization statistics are always
derived from the training pairs of the temporal split only; test and rollout
splits reuse them via ``set_normalization_params`` instead of recomputing.
"""

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.spatial import KDTree
from torch.utils.data import Dataset

from .temporal import (
    TemporalSplit,
    compute_global_normalization_params,
    copy_normalization_params,
    make_temporal_split,
)

if TYPE_CHECKING:
    from ..data_processor.mesh_quad import QuadTreeMesh

NormalizationParams = dict[str, np.ndarray]


class TemporalPairMixin(Dataset):
    """Split/normalization plumbing shared by all four benchmark datasets.

    Subclasses load ``self.coords``/``self.flows`` frame arrays, set
    ``self.split`` and ``self.normalize``, call :meth:`_init_temporal_pair`,
    then :meth:`_split_data`. The protocol invariant: normalization statistics
    come from the training pairs only; other splits reuse them.
    """

    # Attributes provided/consumed by concrete subclasses.
    normalize: bool
    split: str
    step_size: int
    train_ratio: float
    split_seed: int
    rollout_holdout_steps: int
    pair_indices: np.ndarray
    temporal_split: TemporalSplit
    coords: np.ndarray
    flows: np.ndarray
    _all_coords: np.ndarray
    _all_flows: np.ndarray
    _original_n_timesteps: int
    _normalization_params: NormalizationParams | None
    _defer_normalization: bool
    coord_mean: np.ndarray | None
    coord_std: np.ndarray | None
    flow_mean: np.ndarray | None
    flow_std: np.ndarray | None

    def _init_temporal_pair(
        self,
        step_size: int,
        train_ratio: float,
        split_seed: int,
        rollout_holdout_steps: int,
        normalization_params: NormalizationParams | None,
        defer_normalization: bool = False,
    ) -> None:
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.split_seed = split_seed
        self.rollout_holdout_steps = rollout_holdout_steps
        self._normalization_params = copy_normalization_params(normalization_params)
        self._defer_normalization = defer_normalization

    def _split_data(self) -> None:
        """Build the temporal split once and expose the requested pair subset."""
        if not hasattr(self, '_all_coords'):
            self._all_coords = self.coords
            self._all_flows = self.flows
            self._original_n_timesteps = len(self._all_coords)
            self.temporal_split = make_temporal_split(
                self._original_n_timesteps,
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
            # Inference/demo-only view over every pair, including the rollout
            # window; statistics must never be derived from it.
            if self.normalize and self._normalization_params is None:
                raise ValueError(
                    "split='all' requires explicitly supplied normalization "
                    "statistics computed on a training split"
                )
            self.pair_indices = np.arange(
                max(0, self._original_n_timesteps - self.step_size), dtype=np.int64
            )
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.coords = self._all_coords
        self.flows = self._all_flows
        # Both spellings stay in sync across the irregular/patch interfaces.
        self.n_timesteps = self._original_n_timesteps
        self.num_timesteps = self._original_n_timesteps

    def set_split(self, split: str) -> None:
        """Expose a different pair subset ('train' | 'test' | 'all')."""
        self.split = split
        self._split_data()

    def clone_for_split(self, split: str) -> 'TemporalPairMixin':
        """Return a shallow copy that exposes a different pair subset."""
        dataset = copy.copy(self)
        dataset.set_split(split)
        return dataset

    def set_normalization_params(self, params: NormalizationParams) -> None:
        """Install training-split statistics and refresh derived values."""
        self._normalization_params = copy_normalization_params(params)
        self._compute_normalization_params()

    def _compute_normalization_params(self) -> None:
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
        assert self.coord_mean is not None and self.coord_std is not None
        return (coords - self.coord_mean) / self.coord_std

    def _normalize_flows(self, flows: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return flows
        assert self.flow_mean is not None and self.flow_std is not None
        return (flows - self.flow_mean) / self.flow_std

    def __len__(self) -> int:
        return len(self.pair_indices)

    def get_normalization_params(self) -> dict[str, Any]:
        """Return a copy of the current Z-score statistics (None if disabled)."""
        if not self.normalize:
            return {
                'coord_mean': None,
                'coord_std': None,
                'flow_mean': None,
                'flow_std': None,
            }
        assert self.coord_mean is not None and self.coord_std is not None
        assert self.flow_mean is not None and self.flow_std is not None
        return {
            'coord_mean': np.asarray(self.coord_mean, dtype=np.float32).copy(),
            'coord_std': np.asarray(self.coord_std, dtype=np.float32).copy(),
            'flow_mean': np.asarray(self.flow_mean, dtype=np.float32).copy(),
            'flow_std': np.asarray(self.flow_std, dtype=np.float32).copy(),
        }

    def get_rollout_sequence(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the reserved contiguous rollout window (coords, flows)."""
        indices = self.temporal_split.rollout_frames
        return self._all_coords[indices], self._all_flows[indices]


class IrregularPairDataset(TemporalPairMixin):
    """An irregular point-cloud dataset returning ``(pos, fx, y)`` pairs."""

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_idx = int(self.pair_indices[idx])
        coords_t = self._all_coords[time_idx]
        flow_t = self._all_flows[time_idx]
        flow_tp1 = self._all_flows[time_idx + self.step_size]

        pos = torch.from_numpy(self._normalize_coords(coords_t)).float()
        fx = torch.from_numpy(self._normalize_flows(flow_t)).float()
        y = torch.from_numpy(self._normalize_flows(flow_tp1)).float()

        return pos, fx, y


class PatchPairDataset(TemporalPairMixin):
    """A patch-tokenized dataset with a global k-NN full-point recovery map."""

    quadtree: 'QuadTreeMesh'
    num_patches: int
    max_points: int
    include_coordinates: bool
    input_dim: int
    output_dim: int
    num_points: int
    sampled_indices: torch.Tensor
    recovery_indices: torch.Tensor
    recovery_weights: torch.Tensor

    def _build_recovery_map(self) -> None:
        """Precompute the k-NN map from sampled patch points to full points."""
        sampled = np.concatenate([patch.points for patch in self.quadtree.patches])
        sampled = np.unique(sampled).astype(np.int64)
        if len(sampled) < 4:
            raise ValueError("APP recovery requires at least four sampled points")

        full_coords = self._all_coords[0]
        knn_index = KDTree(full_coords[sampled])
        # pi-lens-ignore: python-sql-injection (scipy KD-tree nearest-neighbor query)
        distances, compact_neighbors = knn_index.query(full_coords, k=4)
        weights = 1.0 / np.maximum(distances, 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)

        self.sampled_indices = torch.from_numpy(sampled).long()
        self.recovery_indices = torch.from_numpy(sampled[compact_neighbors]).long()
        self.recovery_weights = torch.from_numpy(weights.astype(np.float32))

    def _build_patch_index(self) -> None:
        """Flatten the quadtree patches into padded index/mask tensors."""
        self._patch_indices = np.zeros(
            (self.num_patches, self.max_points), dtype=np.int64
        )
        self._patch_mask = np.zeros(
            (self.num_patches, self.max_points), dtype=bool
        )
        for patch_idx, patch in enumerate(self.quadtree.patches):
            n_points = len(patch.points)
            self._patch_indices[patch_idx, :n_points] = patch.points
            self._patch_mask[patch_idx, :n_points] = True
        self._mask_tensor = torch.from_numpy(self._patch_mask)

    def _extract_patches(self, coords: np.ndarray, flows: np.ndarray) -> torch.Tensor:
        values = (
            np.concatenate([coords, flows], axis=1)
            if self.include_coordinates else flows
        )
        patches = values[self._patch_indices]
        patches[~self._patch_mask] = 0.0
        return torch.from_numpy(patches.reshape(self.num_patches, -1))

    def _extract_flow_patches(self, flows: np.ndarray) -> torch.Tensor:
        patches = flows[self._patch_indices]
        patches[~self._patch_mask] = 0.0
        return torch.from_numpy(patches.reshape(self.num_patches, -1))

    def _create_mask(self) -> torch.Tensor:
        return self._mask_tensor

    def get_quadtree(self) -> 'QuadTreeMesh':
        """Return the quadtree partition built on the first frame."""
        return self.quadtree


class MultiConditionDatasetMixin(Dataset):
    """(condition_id, local_id) indexing shared by multi-condition wrappers."""

    sub_datasets: list[Any]
    _index_map: list[tuple[int, int]]
    normalization_params: NormalizationParams | None

    def _build_index_map(self) -> None:
        """Flatten the sub-datasets into (condition_id, local_idx) pairs."""
        self._index_map = []
        for cond_id, ds in enumerate(self.sub_datasets):
            for local_idx in range(len(ds)):
                self._index_map.append((cond_id, local_idx))

    def __len__(self) -> int:
        return len(self._index_map)

    def get_sub_dataset(self, condition_id: int) -> Any:
        return self.sub_datasets[condition_id]

    def get_normalization_params(self) -> NormalizationParams | None:
        return copy_normalization_params(self.normalization_params)

    @classmethod
    def from_existing(cls, source: Any, split: str) -> Any:
        """Return a shallow copy of ``source`` exposing a different split."""
        dataset = copy.copy(source)
        dataset.sub_datasets = [
            sub_dataset.clone_for_split(split)
            for sub_dataset in source.sub_datasets
        ]
        dataset._build_index_map()
        return dataset


class MultiConditionIrregularDatasetMixin(MultiConditionDatasetMixin):
    """Multi-condition wrapper returning ``(pos, fx, y, condition_id)``."""

    n_channels: int
    global_max_points: int

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_id, local_idx = self._index_map[idx]
        pos, fx, y = self.sub_datasets[cond_id][local_idx]
        return pos, fx, y, torch.tensor(cond_id, dtype=torch.long)

    def get_global_shape(self) -> dict[str, int]:
        return {
            'n_channels': self.n_channels,
            'max_points': self.global_max_points,
        }


class MultiConditionPatchDatasetMixin(MultiConditionDatasetMixin):
    """Multi-condition wrapper padding patch samples to a global shape."""

    input_dim: int
    output_dim: int
    global_max_patches: int
    global_max_points: int
    global_max_full_points: int

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cond_id, local_idx = self._index_map[idx]
        sample: dict[str, torch.Tensor] = self.sub_datasets[cond_id][local_idx]

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
        p = orig_input.shape[0]
        n = orig_mask.shape[1]
        p_use = min(p, self.global_max_patches)
        n_use = min(n, self.global_max_points)

        input_padded[:p_use, :n_use * self.input_dim] = orig_input[:p_use, :n_use * self.input_dim]
        output_padded[:p_use, :n_use * self.output_dim] = orig_output[:p_use, :n_use * self.output_dim]
        mask_padded[:p_use, :n_use] = orig_mask[:p_use, :n_use]
        result: dict[str, torch.Tensor] = {
            'input': input_padded,
            'output': output_padded,
            'mask': mask_padded,
            'condition_id': torch.tensor(cond_id, dtype=torch.long),
            'time_index': sample['time_index'],
        }
        if 'full_target' in sample:
            full_target_padded = torch.zeros(
                (self.global_max_full_points, self.output_dim), dtype=torch.float32
            )
            full_point_mask = torch.zeros(
                self.global_max_full_points, dtype=torch.bool
            )
            n_full = min(sample['full_target'].shape[0], self.global_max_full_points)
            full_target_padded[:n_full] = sample['full_target'][:n_full]
            full_point_mask[:n_full] = True
            result['full_target'] = full_target_padded
            result['full_point_mask'] = full_point_mask

        if 'params_embedding' in sample:
            result['params_embedding'] = sample['params_embedding']
        if 'params_text' in sample:
            result['params_text'] = sample['params_text']

        return result

    def get_global_shape(self) -> dict[str, int]:
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_patches': self.global_max_patches,
            'max_points': self.global_max_points,
            'full_points': self.global_max_full_points,
        }
