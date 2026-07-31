import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rebuild_ship_flow_cache import rebuild_condition


def test_rebuild_cache_uses_numeric_order_and_fixed_reference(tmp_path: Path):
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float32)
    for timestep in range(12):
        shifted = points + np.array([0.001 * timestep, 0.0, 0.0])
        frame = pd.DataFrame({
            'Center:0': shifted[:, 0],
            'Center:1': shifted[:, 1],
            'Center:2': shifted[:, 2],
            'U:0': np.full(len(points), timestep),
            'U:1': np.full(len(points), timestep + 1),
            'U:2': np.full(len(points), timestep + 2),
            'p_rgh': np.full(len(points), timestep + 3),
        })
        frame.to_csv(tmp_path / f'timestep_{timestep}.csv', index=False)

    np.savez(
        tmp_path / 'flow_cache.npz',
        coords=np.zeros((1, 1, 2), dtype=np.float32),
        flows=np.zeros((1, 1, 4), dtype=np.float32),
        channels=np.array(['U:0', 'U:1', 'U:2', 'p_rgh']),
    )

    rebuild_condition(
        tmp_path, 'flow_cache.npz', 'flow_cache_lexicographic_backup.npz', k=4
    )

    assert (tmp_path / 'flow_cache_lexicographic_backup.npz').exists()
    cache = np.load(tmp_path / 'flow_cache.npz')
    assert cache['frame_indices'].tolist() == list(range(12))
    assert np.array_equal(cache['coords'][0], cache['coords'][-1])
    assert np.allclose(cache['flows'][10, :, 0], 10.0)

    report = json.loads((tmp_path / 'flow_cache_alignment_report.json').read_text())
    assert report['reference_points'] == len(points)
    assert report['frame_index_end'] == 11
