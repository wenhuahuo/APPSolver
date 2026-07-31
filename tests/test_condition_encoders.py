from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.datasets.shipBench import MultiConditionPatchDataset
from src.models.patch import DPT, Transformer


def _write_condition(path: Path, values):
    path.mkdir(parents=True)
    side = 5
    x, y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    coords0 = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)
    coords = np.repeat(coords0[None], 20, axis=0)
    flows = np.zeros((20, len(coords0), 4), dtype=np.float32)
    np.savez(
        path / 'flow_cache.npz',
        coords=coords,
        flows=flows,
        channels=np.array(['U:0', 'U:1', 'U:2', 'p_rgh']),
        frame_indices=np.arange(20),
    )
    with open(path / 'ship_params_test.yaml', 'w', encoding='utf-8') as handle:
        yaml.safe_dump({'geometry': {'a': values[0]}, 'case': {'speed': values[1]}}, handle)


@pytest.mark.parametrize('encoder', ['mlp', 'fourier_mlp', 'film'])
def test_lightweight_condition_encoders_transformer(encoder):
    model = Transformer(
        in_flattened_dim=12,
        out_flattened_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        params_dim=5,
        condition_encoder=encoder,
    )
    output = model(torch.randn(2, 7, 12), torch.randn(2, 5))
    assert output.shape == (2, 7, 8)
    output.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize('encoder', ['mlp', 'fourier_mlp', 'film'])
def test_lightweight_condition_encoders_dpt(encoder):
    model = DPT(
        in_flattened_dim=12,
        out_flattened_dim=8,
        features=16,
        d_model=16,
        n_heads=4,
        n_layers=4,
        params_dim=5,
        condition_encoder=encoder,
        max_patches=32,
    )
    output = model(torch.randn(2, 16, 12), params_embed=torch.randn(2, 5))
    assert output.shape == (2, 16, 8)


def test_numeric_conditions_use_training_statistics_for_heldout_data(tmp_path):
    train_a = tmp_path / 'train_a'
    train_b = tmp_path / 'train_b'
    heldout = tmp_path / 'heldout'
    _write_condition(train_a, (0.0, 1.0))
    _write_condition(train_b, (2.0, 3.0))
    _write_condition(heldout, (4.0, 5.0))

    train = MultiConditionPatchDataset(
        [str(train_a), str(train_b)],
        patch_size=8,
        enable_params=True,
        embedding_mode='numeric',
        rollout_holdout_steps=0,
    )
    condition_stats = train.get_condition_normalization_params()
    validation = MultiConditionPatchDataset(
        [str(heldout)],
        patch_size=8,
        enable_params=True,
        embedding_mode='numeric',
        rollout_holdout_steps=0,
        normalization_params=train.get_normalization_params(),
        condition_normalization_params=condition_stats,
    )

    assert train.embedding_dim == 2
    assert train.sub_datasets[0].params_embedding.tolist() == pytest.approx([-1.0, -1.0])
    assert train.sub_datasets[1].params_embedding.tolist() == pytest.approx([1.0, 1.0])
    assert validation.sub_datasets[0].params_embedding.tolist() == pytest.approx([3.0, 3.0])
