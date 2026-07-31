"""Quadtree assignment utilities for controlled tokenizer comparisons."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .mesh_quad import QuadTreeMesh


@dataclass(frozen=True)
class PartitionSpec:
    token_ids: np.ndarray
    token_centers: np.ndarray
    num_tokens: int
    patch_capacity: int
    adaptive: bool


def build_partition(
    coords: np.ndarray,
    patch_capacity: int,
    adaptive: bool,
    ship_length: float = 7.0,
    ref_point: Tuple[float, float] = (3.0, 0.0),
    distance_threshold_1: float = 1.0,
    distance_threshold_2: float = 1.5,
) -> PartitionSpec:
    """Build a full-point quadtree assignment without spatial downsampling."""
    tree = QuadTreeMesh(
        np.asarray(coords, dtype=np.float32),
        patch_size=int(patch_capacity),
        ship_length=ship_length,
        ref_point=ref_point,
        distance_threshold_1=distance_threshold_1,
        distance_threshold_2=distance_threshold_2,
        enable_distance_refine=adaptive,
    )
    token_ids = np.full(len(coords), -1, dtype=np.int64)
    token_centers = np.empty((len(tree.patches), 2), dtype=np.float32)
    for token_id, patch in enumerate(tree.patches):
        token_ids[patch.points] = token_id
        token_centers[token_id] = np.asarray(coords)[patch.points].mean(axis=0)

    if np.any(token_ids < 0):
        raise RuntimeError("Quadtree partition did not assign every reference point")

    return PartitionSpec(
        token_ids=token_ids,
        token_centers=token_centers,
        num_tokens=len(tree.patches),
        patch_capacity=int(patch_capacity),
        adaptive=bool(adaptive),
    )


def match_partition_budget(
    coords: np.ndarray,
    target_tokens: int,
    adaptive: bool,
    ship_length: float = 7.0,
    ref_point: Tuple[float, float] = (3.0, 0.0),
    distance_threshold_1: float = 1.0,
    distance_threshold_2: float = 1.5,
    search_radius: int = 16,
) -> PartitionSpec:
    """Find the patch capacity whose leaf count is closest to ``target_tokens``."""
    if target_tokens < 2:
        raise ValueError("target_tokens must be at least 2")

    kwargs = dict(
        coords=coords,
        adaptive=adaptive,
        ship_length=ship_length,
        ref_point=ref_point,
        distance_threshold_1=distance_threshold_1,
        distance_threshold_2=distance_threshold_2,
    )
    cache = {}

    def evaluate(capacity: int) -> PartitionSpec:
        capacity = max(4, int(capacity))
        if capacity not in cache:
            cache[capacity] = build_partition(
                patch_capacity=capacity,
                **kwargs,
            )
        return cache[capacity]

    low = 4
    high = max(4, len(coords) * (4 if adaptive else 1))
    while low <= high:
        capacity = (low + high) // 2
        spec = evaluate(capacity)
        if spec.num_tokens > target_tokens:
            low = capacity + 1
        elif spec.num_tokens < target_tokens:
            high = capacity - 1
        else:
            return spec

    center_candidates = {low, high}
    for center in tuple(center_candidates):
        for capacity in range(center - search_radius, center + search_radius + 1):
            if capacity >= 4:
                evaluate(capacity)

    return min(
        cache.values(),
        key=lambda spec: (
            abs(spec.num_tokens - target_tokens),
            spec.patch_capacity,
        ),
    )
