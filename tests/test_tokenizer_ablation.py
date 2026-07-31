import numpy as np
import torch

from src.data_processor.tokenizer_partition import (
    build_partition,
    match_partition_budget,
)
from src.models.irregular.tokenizer_ablation import PointTokenOperator


def _grid(nx=12, ny=8):
    x, y = np.meshgrid(np.linspace(0, 2, nx), np.linspace(-1, 1, ny))
    return np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)


def test_quadtree_partition_assigns_every_point_and_matches_budget():
    coords = _grid()
    spec = build_partition(coords, patch_capacity=8, adaptive=False)

    assert spec.token_ids.shape == (len(coords),)
    assert spec.token_ids.min() == 0
    assert spec.token_ids.max() == spec.num_tokens - 1
    assert np.array_equal(np.unique(spec.token_ids), np.arange(spec.num_tokens))

    matched = match_partition_budget(
        coords, target_tokens=16, adaptive=False, search_radius=4
    )
    candidates = [
        build_partition(coords, patch_capacity=value, adaptive=False)
        for value in range(max(4, matched.patch_capacity - 4), matched.patch_capacity + 5)
    ]
    assert abs(matched.num_tokens - 16) == min(
        abs(candidate.num_tokens - 16) for candidate in candidates
    )


def test_hard_and_learned_tokenizers_share_shape_and_parameter_budget():
    torch.manual_seed(3)
    positions = torch.randn(2, 20, 2)
    flow = torch.randn(2, 20, 4)
    token_ids = torch.arange(20) % 5
    kwargs = dict(
        input_dim=6, output_dim=4, d_model=16, nhead=4,
        num_layers=2, dim_feedforward=32, max_tokens=8, dropout=0.0,
    )
    hard = PointTokenOperator(tokenizer='hard', **kwargs)
    learned = PointTokenOperator(tokenizer='learned', **kwargs)

    hard_output, hard_diag = hard(
        positions, flow, num_tokens=5, token_ids=token_ids,
        return_diagnostics=True,
    )
    learned_output, learned_diag = learned(
        positions, flow, num_tokens=5, return_diagnostics=True,
    )

    assert hard_output.shape == learned_output.shape == (2, 20, 4)
    assert sum(p.numel() for p in hard.parameters()) == sum(
        p.numel() for p in learned.parameters()
    )
    assert hard_diag['assignment_entropy'].item() == 0.0
    assert 0.0 <= learned_diag['assignment_entropy'].item() <= 1.0
    assert 0.0 <= learned_diag['assignment_confidence'].item() <= 1.0

    learned_output.square().mean().backward()
    assert learned.assignment_key.weight.grad is not None
    assert learned.token_embed.grad is not None


def test_tokenizers_are_point_permutation_equivariant():
    torch.manual_seed(5)
    positions = torch.randn(1, 18, 2)
    flow = torch.randn(1, 18, 4)
    token_ids = torch.arange(18) % 6
    permutation = torch.randperm(18)
    inverse = torch.argsort(permutation)

    for tokenizer in ['hard', 'learned']:
        model = PointTokenOperator(
            input_dim=6, output_dim=4, d_model=16, nhead=4,
            num_layers=1, dim_feedforward=32, max_tokens=8,
            tokenizer=tokenizer, dropout=0.0,
        ).eval()
        kwargs = {'token_ids': token_ids} if tokenizer == 'hard' else {}
        permuted_kwargs = (
            {'token_ids': token_ids[permutation]} if tokenizer == 'hard' else {}
        )
        with torch.no_grad():
            reference = model(positions, flow, 6, **kwargs)
            permuted = model(
                positions[:, permutation], flow[:, permutation], 6,
                **permuted_kwargs,
            )
        assert torch.allclose(reference, permuted[:, inverse], atol=1e-5, rtol=1e-5)
