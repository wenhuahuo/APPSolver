"""Build binary flow caches for CFDBench cases.

Cache format (.npz):
  - coords: (T, N, 2) float32
  - flows: (T, N, C) float32
  - channels: (C,) str
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from fnmatch import fnmatch
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


CFD_CHANNELS = ['y-velocity', 'x-velocity']


CFD_COLUMN_ALIASES = {
    'volume-fraction-water': 'water-vof',
    'y-velocity-water': 'y-velocity',
    'x-velocity-water': 'x-velocity',
}


def _standardize_cfd_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for alias, canonical in CFD_COLUMN_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            rename_map[alias] = canonical
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _align_series(
    coords_list: Sequence[np.ndarray], flows_list: Sequence[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    min_cells = min(len(c) for c in coords_list)
    reference_coords = coords_list[0]

    aligned_coords = []
    aligned_flows = []
    for coords, flows in zip(coords_list, flows_list):
        tree = KDTree(coords)
        _, indices = tree.query(reference_coords[:min_cells])

        aligned_coords.append(coords[indices].astype(np.float32))
        aligned_flows.append(flows[indices].astype(np.float32))

    return (
        np.asarray(aligned_coords, dtype=np.float32),
        np.asarray(aligned_flows, dtype=np.float32),
    )


def _build_cfd_cache(data_dir: str, cache_path: str) -> None:
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith('data') and f.endswith('.txt')
    )
    if not files:
        raise RuntimeError('no data*.txt found')

    coords_list: List[np.ndarray] = []
    flows_list: List[np.ndarray] = []
    for name in files:
        path = os.path.join(data_dir, name)
        df = pd.read_csv(path, sep=r'\s+')
        df = _standardize_cfd_columns(df)

        required = ['x-coordinate', 'y-coordinate', *CFD_CHANNELS]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f'missing columns {missing} in {path}')

        coords = df[['x-coordinate', 'y-coordinate']].values.astype(np.float32)
        flows = df[CFD_CHANNELS].values.astype(np.float32)

        coords_list.append(coords)
        flows_list.append(flows)

    coords, flows = _align_series(coords_list, flows_list)
    np.savez(cache_path, coords=coords, flows=flows, channels=np.asarray(CFD_CHANNELS))


def _discover_cfd_dirs(root: str) -> List[str]:
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(fnmatch(f, 'data*.txt') for f in filenames):
            found.append(dirpath)
    return sorted(found)


def _run_one(task: Tuple[str, str, bool]) -> Tuple[str, str, str]:
    data_dir, cache_filename, overwrite = task
    cache_path = os.path.join(data_dir, cache_filename)

    if (not overwrite) and os.path.exists(cache_path):
        return 'skipped', data_dir, 'cache exists'

    try:
        _build_cfd_cache(data_dir, cache_path)
    except Exception as exc:  # noqa: BLE001
        return 'failed', data_dir, str(exc)

    return 'ok', data_dir, cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Build CFDBench flow caches (.npz)')
    parser.add_argument('--root', type=str, default='datasets/cfdBench')
    parser.add_argument('--cache-filename', type=str, default='flow_cache.npz')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument(
        '--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    args = parser.parse_args()

    tasks: List[Tuple[str, str, bool]] = []
    if os.path.isdir(args.root):
        for data_dir in _discover_cfd_dirs(args.root):
            tasks.append((data_dir, args.cache_filename, args.overwrite))
    else:
        print(f'[WARN] CFDBench root not found: {args.root}')

    if not tasks:
        print('No dataset directories found. Nothing to do.')
        return

    print(f'Found {len(tasks)} directories to process')
    print(f'Workers: {args.workers}')

    ok = 0
    skipped = 0
    failed = 0

    if args.workers <= 1:
        for task in tasks:
            status, data_dir, detail = _run_one(task)
            if status == 'ok':
                ok += 1
                print(f'[OK] {data_dir} -> {detail}')
            elif status == 'skipped':
                skipped += 1
                print(f'[SKIP] {data_dir}: {detail}')
            else:
                failed += 1
                print(f'[FAIL] {data_dir}: {detail}')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_one, t) for t in tasks]
            for future in as_completed(futures):
                status, data_dir, detail = future.result()
                if status == 'ok':
                    ok += 1
                    print(f'[OK] {data_dir} -> {detail}')
                elif status == 'skipped':
                    skipped += 1
                    print(f'[SKIP] {data_dir}: {detail}')
                else:
                    failed += 1
                    print(f'[FAIL] {data_dir}: {detail}')

    print('\nDone')
    print(f'  ok:      {ok}')
    print(f'  skipped: {skipped}')
    print(f'  failed:  {failed}')


if __name__ == '__main__':
    main()
