"""Build geometry-bound PCNO auxiliary caches for CFDBench cases."""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.models.irregular.pcno import build_aux_from_pos, save_pcno_aux_cache


def _discover_cases(root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in root.glob('case*/flow_cache.npz')
        if path.is_file()
    )


def _build_one(task: tuple[str, str, int, int, bool]) -> tuple[str, str, str]:
    case_dir_str, cache_filename, k_neighbors, nmeasures, overwrite = task
    case_dir = Path(case_dir_str)
    output_path = case_dir / cache_filename
    if output_path.exists() and not overwrite:
        return 'skipped', case_dir_str, 'cache exists'

    flow_cache = case_dir / 'flow_cache.npz'
    try:
        with np.load(flow_cache, allow_pickle=False) as data:
            if 'coords' not in data.files:
                raise ValueError(f'missing coords in {flow_cache}')
            reference = np.asarray(data['coords'][0], dtype=np.float32).copy()
        aux = build_aux_from_pos(
            reference, k_neighbors=k_neighbors, nmeasures=nmeasures,
        )
        save_pcno_aux_cache(
            str(output_path), aux, reference, k_neighbors, nmeasures,
        )
    except Exception as exc:  # noqa: BLE001
        return 'failed', case_dir_str, str(exc)
    return 'ok', case_dir_str, str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfd-root', required=True, type=Path)
    parser.add_argument('--cache-filename', default='pcno_aux_k8_m1.npz')
    parser.add_argument('--k-neighbors', type=int, default=8)
    parser.add_argument('--nmeasures', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cases = _discover_cases(args.cfd_root)
    if not cases:
        raise SystemExit(f'No case*/flow_cache.npz found under {args.cfd_root}')

    tasks = [
        (
            str(case), args.cache_filename, args.k_neighbors,
            args.nmeasures, args.overwrite,
        )
        for case in cases
    ]
    counts = {'ok': 0, 'skipped': 0, 'failed': 0}
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_build_one, task) for task in tasks]
        for future in as_completed(futures):
            status, case_dir, detail = future.result()
            counts[status] += 1
            print(f'[{status}] {case_dir}: {detail}', flush=True)
            if status == 'failed':
                failures.append((case_dir, detail))

    print(
        f"Summary: total={len(tasks)} ok={counts['ok']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
