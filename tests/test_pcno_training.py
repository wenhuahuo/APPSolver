from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.train_irregular_pcno import (
    PCNODataset,
    expand_aux,
    move_aux_to_device,
    pcno_collate_fn,
)
from src.models.irregular.pcno import (
    build_aux_from_pos,
    load_pcno_aux_cache,
    save_pcno_aux_cache,
)


class _PointDataset(Dataset):
    def __init__(self):
        x, y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
        coords = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)
        self.coords = np.repeat(coords[None], 3, axis=0)
        self.flows = np.zeros((3, len(coords), 4), dtype=np.float32)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        pos = torch.from_numpy(self.coords[index])
        flow = torch.from_numpy(self.flows[index])
        target = torch.from_numpy(self.flows[index + 1])
        return pos, flow, target


def _reference_gradient_weights(pos, neighbor_indices):
    weights = []
    for i in range(len(pos)):
        neighbors = neighbor_indices[i]
        dx = pos[neighbors] - pos[i]
        u, singular_values, vt = np.linalg.svd(dx, full_matrices=False)
        rcond = 1e-3 * singular_values[0] if singular_values[0] > 0 else 1e-12
        inverse = np.where(singular_values > rcond, 1.0 / singular_values, 0.0)
        pseudo_inverse = (vt.T * inverse) @ u.T
        for column in range(len(neighbors)):
            weights.append(pseudo_inverse[:, column])
    return np.asarray(weights, dtype=np.float32)


def test_vectorized_pcno_aux_matches_per_node_svd():
    rng = np.random.default_rng(42)
    pos = rng.normal(size=(25, 2)).astype(np.float32)
    aux = build_aux_from_pos(pos, k_neighbors=4)
    edges = aux['directed_edges']
    neighbors = edges[:, 1].reshape(len(pos), 4)
    expected = _reference_gradient_weights(pos, neighbors)

    np.testing.assert_allclose(aux['edge_gradient_weights'], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(edges[:, 0], np.repeat(np.arange(len(pos)), 4))


def test_pcno_aux_cache_is_bound_to_reference_geometry(tmp_path: Path):
    rng = np.random.default_rng(7)
    pos = rng.normal(size=(20, 2)).astype(np.float32)
    aux = build_aux_from_pos(pos, k_neighbors=4)
    cache_path = tmp_path / 'pcno_aux.npz'
    save_pcno_aux_cache(str(cache_path), aux, pos, 4, 1)

    loaded = load_pcno_aux_cache(str(cache_path), pos, 4, 1)
    for key in aux:
        np.testing.assert_array_equal(loaded[key], aux[key])

    changed = pos.copy()
    changed[0, 0] += 1
    try:
        load_pcno_aux_cache(str(cache_path), changed, 4, 1)
    except ValueError as exc:
        assert 'geometry mismatch' in str(exc)
    else:
        raise AssertionError('geometry mismatch was not rejected')


def test_pcno_geometry_is_shared_and_expanded_per_batch():
    dataset = PCNODataset(_PointDataset(), k_neighbors=4)
    loader = DataLoader(dataset, batch_size=2, collate_fn=pcno_collate_fn)
    pos, flow, target, condition_id = next(iter(loader))

    assert condition_id == 0
    assert pos.shape == flow.shape[:2] + (2,)
    assert target.shape == flow.shape

    device_aux = move_aux_to_device(dataset.aux_by_condition, torch.device('cpu'))
    batch_aux = expand_aux(device_aux[condition_id], pos.size(0))
    assert batch_aux['node_weights'].shape[0] == 2
    assert batch_aux['directed_edges'].shape[0] == 2
    assert batch_aux['directed_edges'].data_ptr() == device_aux[0]['directed_edges'].data_ptr()
