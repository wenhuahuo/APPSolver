"""Rebuild ShipBench caches in numeric time order on a fixed reference point set.

For each condition, the first numerically ordered frame defines the fixed
reference free-surface points. Flow values from every later frame are
interpolated to those points using 3-D inverse-distance weighted k-NN. The
model-facing coordinates remain the reference (x, y) coordinates.

The existing flow_cache.npz is backed up before the rebuilt cache replaces it.
"""

import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


FLOW_CHANNELS = ['U:0', 'U:1', 'U:2', 'p_rgh']
COORD_CHANNELS = ['Center:0', 'Center:1', 'Center:2']
_TIMESTEP_RE = re.compile(r'^timestep_(\d+)\.csv$')


def timestep_index(path: Path) -> int:
    match = _TIMESTEP_RE.match(path.name)
    if match is None:
        raise ValueError(f'Unexpected timestep filename: {path.name}')
    return int(match.group(1))


def discover_condition_dirs(root: Path):
    return sorted({path.parent for path in root.rglob('timestep_*.csv')})


def ordered_timestep_files(condition_dir: Path):
    files = sorted(condition_dir.glob('timestep_*.csv'), key=timestep_index)
    indices = [timestep_index(path) for path in files]
    if not files:
        raise ValueError(f'No timestep CSV files found in {condition_dir}')
    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise ValueError(
            f'Timestep indices in {condition_dir} are not contiguous: '
            f'{indices[0]}..{indices[-1]}'
        )
    return files, np.asarray(indices, dtype=np.int64)


def load_frame(path: Path):
    columns = [*COORD_CHANNELS, *FLOW_CHANNELS]
    frame = pd.read_csv(path, usecols=columns)
    coords = frame[COORD_CHANNELS].to_numpy(dtype=np.float32)
    flows = frame[FLOW_CHANNELS].to_numpy(dtype=np.float32)
    return coords, flows


def interpolate_to_reference(
    reference, source_coords, source_flows, k=4, workers=1
):
    if len(source_coords) < k:
        raise ValueError(f'Need at least {k} source points, got {len(source_coords)}')

    distances, neighbors = cKDTree(source_coords).query(
        reference, k=k, workers=workers
    )
    aligned = np.empty((len(reference), source_flows.shape[1]), dtype=np.float32)
    exact = distances[:, 0] <= 1e-12
    aligned[exact] = source_flows[neighbors[exact, 0]]

    non_exact = ~exact
    weights = 1.0 / np.maximum(distances[non_exact], 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)
    values = source_flows[neighbors[non_exact]]
    aligned[non_exact] = np.sum(values * weights[..., None], axis=1)
    return aligned, distances[:, 0]


def rebuild_condition(
    condition_dir: Path, cache_name: str, backup_name: str, k: int,
    workers: int = 1,
):
    files, frame_indices = ordered_timestep_files(condition_dir)
    reference, first_flow = load_frame(files[0])
    n_frames = len(files)
    n_reference = len(reference)

    aligned_flows = np.empty(
        (n_frames, n_reference, len(FLOW_CHANNELS)), dtype=np.float32
    )
    aligned_flows[0] = first_flow
    source_counts = np.empty(n_frames, dtype=np.int64)
    source_counts[0] = n_reference
    distance_mean = np.zeros(n_frames, dtype=np.float64)
    distance_p95 = np.zeros(n_frames, dtype=np.float64)
    distance_p99 = np.zeros(n_frames, dtype=np.float64)
    distance_max = np.zeros(n_frames, dtype=np.float64)

    started = time.time()
    for frame_id, path in enumerate(files[1:], start=1):
        source_coords, source_flows = load_frame(path)
        aligned, nearest_distance = interpolate_to_reference(
            reference, source_coords, source_flows, k=k, workers=workers
        )
        aligned_flows[frame_id] = aligned
        source_counts[frame_id] = len(source_coords)
        distance_mean[frame_id] = float(nearest_distance.mean())
        distance_p95[frame_id] = float(np.quantile(nearest_distance, 0.95))
        distance_p99[frame_id] = float(np.quantile(nearest_distance, 0.99))
        distance_max[frame_id] = float(nearest_distance.max())

        if frame_id % 50 == 0 or frame_id == n_frames - 1:
            print(f'  {condition_dir}: {frame_id + 1}/{n_frames} frames')

    cache_path = condition_dir / cache_name
    backup_path = condition_dir / backup_name
    temp_path = condition_dir / f'.{cache_name}.tmp'
    report_path = condition_dir / 'flow_cache_alignment_report.json'

    if backup_path.exists():
        raise FileExistsError(f'Backup already exists: {backup_path}')
    if temp_path.exists():
        temp_path.unlink()

    fixed_coords = np.broadcast_to(
        reference[None, :, :2], (n_frames, n_reference, 2)
    )
    with open(temp_path, 'wb') as handle:
        np.savez(
            handle,
            coords=fixed_coords,
            flows=aligned_flows,
            channels=np.asarray(FLOW_CHANNELS),
            frame_indices=frame_indices,
            reference_coords_3d=reference,
            source_point_counts=source_counts,
            alignment_distance_mean=distance_mean,
            alignment_distance_p95=distance_p95,
            alignment_distance_p99=distance_p99,
            alignment_distance_max=distance_max,
        )

    if cache_path.exists():
        os.replace(cache_path, backup_path)
    os.replace(temp_path, cache_path)

    report = {
        'condition_dir': str(condition_dir),
        'cache': str(cache_path),
        'backup': str(backup_path) if backup_path.exists() else None,
        'frames': n_frames,
        'frame_index_start': int(frame_indices[0]),
        'frame_index_end': int(frame_indices[-1]),
        'reference_points': n_reference,
        'source_point_count_min': int(source_counts.min()),
        'source_point_count_max': int(source_counts.max()),
        'interpolation': f'3-D inverse-distance weighted k-NN (k={k})',
        'nearest_distance_mean_max': float(distance_mean.max()),
        'nearest_distance_p95_max': float(distance_p95.max()),
        'nearest_distance_p99_max': float(distance_p99.max()),
        'nearest_distance_max': float(distance_max.max()),
        'elapsed_seconds': time.time() - started,
    }
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild ShipBench caches with numeric ordering and fixed points'
    )
    parser.add_argument('--root', type=Path, default=Path('datasets/shipBench'))
    parser.add_argument('--condition_dirs', nargs='*', type=Path)
    parser.add_argument('--cache_name', default='flow_cache.npz')
    parser.add_argument('--backup_name', default='flow_cache_lexicographic_backup.npz')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    condition_dirs = args.condition_dirs or discover_condition_dirs(args.root)
    if not condition_dirs:
        raise ValueError(f'No ShipBench condition directories found under {args.root}')

    reports = []
    for condition_dir in condition_dirs:
        print(f'\nRebuilding {condition_dir}')
        reports.append(
            rebuild_condition(
                condition_dir, args.cache_name, args.backup_name, args.k,
                workers=args.workers,
            )
        )

    print('\nCompleted cache rebuild:')
    for report in reports:
        print(
            f"  {report['condition_dir']}: {report['frames']} frames, "
            f"N={report['reference_points']}, "
            f"max p99 distance={report['nearest_distance_p99_max']:.6g}"
        )


if __name__ == '__main__':
    main()
