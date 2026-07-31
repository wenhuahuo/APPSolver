from pathlib import Path

import json
import numpy as np

from scripts.evaluate_persistence import main


def _write_cache(path: Path, offset: float):
    path.mkdir(parents=True)
    x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    coords0 = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)
    coords = np.repeat(coords0[None], 70, axis=0)
    time = np.arange(70, dtype=np.float32)[:, None, None]
    channels = np.arange(4, dtype=np.float32)[None, None, :]
    flows = offset + time + channels + np.zeros((1, len(coords0), 1), dtype=np.float32)
    np.savez(
        path / 'flow_cache.npz', coords=coords, flows=flows,
        channels=np.array(['U:0', 'U:1', 'U:2', 'p_rgh']),
        frame_indices=np.arange(len(coords)),
    )


def test_persistence_cli_uses_corrected_pairs_and_writes_metrics(tmp_path, monkeypatch):
    case_a = tmp_path / '1Re'
    case_b = tmp_path / '2Re'
    output = tmp_path / 'output'
    _write_cache(case_a, 0.0)
    _write_cache(case_b, 10.0)

    monkeypatch.setattr(
        'sys.argv',
        [
            'evaluate_persistence.py', '--data_dirs', str(case_a), str(case_b),
            '--rollout_holdout_steps', '10', '--save_dir', str(output),
        ],
    )
    main()

    with open(output / 'metrics.json', encoding='utf-8') as handle:
        metrics = json.load(handle)
    assert metrics['overall']['mae'] > 0
    assert len(metrics['overall']['mae_per_channel']) == 4
    assert (output / 'data_protocol.json').exists()
    assert (output / 'normalization_stats.npz').exists()
