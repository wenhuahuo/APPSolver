"""Build the E03 per-seed and multi-seed condition-encoder tables."""

import argparse
import csv
import json
import statistics
from pathlib import Path


SCENARIOS = [
    ('full', 'full'),
    ('val_DTC', 'leave_out_DTC'),
    ('val_KCS', 'leave_out_KCS'),
    ('val_KVLCC2', 'leave_out_KVLCC2'),
]
MODELS = [('transformer', 'APP-Transformer'), ('dpt', 'APP-DPT')]
CONDITIONERS = [
    ('zero', 'zero'),
    ('qwen25', 'qwen2.5'),
    ('qwen35', 'qwen3.5'),
    ('mlp', 'mlp'),
    ('fourier_mlp', 'fourier_mlp'),
    ('film', 'film'),
]
RUN_FIELDS = [
    'dataset', 'scenario', 'model', 'condition_encoder', 'seed', 'split_seed',
    'checkpoint', 'selection_metric', 'best_step', 'final_step',
    'best_is_final_step', 'mae', 'mse', 'rmse', 'relative_l2', 'mae_u',
    'mae_v', 'mae_w', 'mae_p_rgh', 'model_parameters',
    'trainable_parameters', 'training_time_to_best_sec',
    'training_time_total_sec', 'source_path', 'best_checkpoint_remote_path',
]
SUMMARY_METRICS = [
    'best_step', 'mae', 'mse', 'rmse', 'relative_l2',
    'training_time_to_best_sec', 'training_time_total_sec',
]
SUMMARY_FIELDS = [
    'dataset', 'scenario', 'model', 'condition_encoder', 'checkpoint',
    'selection_metric', 'seed_count', 'seeds', 'model_parameters',
    'trainable_parameters',
]
for metric in SUMMARY_METRICS:
    SUMMARY_FIELDS.extend((f'{metric}_mean', f'{metric}_std'))
SUMMARY_FIELDS.insert(12, 'best_is_final_count')


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_root', default='outputs/rebuttal_r5_condition_encoders'
    )
    parser.add_argument(
        '--output_dir',
        default='docs/NeuripsRebuttal/experiments/exp03_condition_encoders',
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--split_seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(args.run_root)
    rows = []
    for scenario_dir, scenario_name in SCENARIOS:
        for model_dir, model_name in MODELS:
            for conditioner_dir, conditioner_name in CONDITIONERS:
                for seed in args.seeds:
                    run_dir = (
                        root / scenario_dir / model_dir / conditioner_dir
                        / f'seed{seed}'
                    )
                    metrics_path = run_dir / 'metrics.json'
                    config_path = run_dir / 'run_config.json'
                    if not metrics_path.is_file() or not config_path.is_file():
                        raise FileNotFoundError(f'Incomplete run: {run_dir}')

                    records = json.loads(metrics_path.read_text())
                    config = json.loads(config_path.read_text())
                    if int(config['seed']) != seed:
                        raise ValueError(f'Unexpected model seed in {config_path}')
                    resolved_split_seed = config.get('resolved_split_seed')
                    if (
                        resolved_split_seed is not None
                        and int(resolved_split_seed) != args.split_seed
                    ):
                        raise ValueError(f'Unexpected split seed in {config_path}')

                    best = min(records, key=lambda record: (record['mae'], record['step']))
                    final = max(records, key=lambda record: record['step'])
                    channels = best['mae_per_channel']
                    source_path = (
                        'docs/NeuripsRebuttal/archive/2026-07-27_pre_reorganization/'
                        f'R5_condition_encoders/runs/{scenario_dir}/{model_dir}/'
                        f'{conditioner_dir}/seed{seed}/metrics.json'
                        if seed == 42 else
                        f'outputs/rebuttal_r5_condition_encoders/{scenario_dir}/'
                        f'{model_dir}/{conditioner_dir}/seed{seed}/metrics.json'
                    )
                    remote_dir = (
                        '/data/home/wenhua_huo/Documents/APPSolver/outputs/'
                        f'rebuttal_r5_condition_encoders/{scenario_dir}/{model_dir}/'
                        f'{conditioner_dir}/seed{seed}'
                    )
                    rows.append({
                        'dataset': 'ShipBench',
                        'scenario': scenario_name,
                        'model': model_name,
                        'condition_encoder': conditioner_name,
                        'seed': seed,
                        'split_seed': args.split_seed,
                        'checkpoint': 'best_validation_mae',
                        'selection_metric': 'mae',
                        'best_step': best['step'],
                        'final_step': final['step'],
                        'best_is_final_step': best['step'] == final['step'],
                        'mae': best['mae'],
                        'mse': best['mse'],
                        'rmse': best['rmse'],
                        'relative_l2': best['relative_l2'],
                        'mae_u': channels[0],
                        'mae_v': channels[1],
                        'mae_w': channels[2],
                        'mae_p_rgh': channels[3],
                        'model_parameters': config['model_params'],
                        'trainable_parameters': config['trainable_params'],
                        'training_time_to_best_sec': best['elapsed_sec'],
                        'training_time_total_sec': final['elapsed_sec'],
                        'source_path': source_path,
                        'best_checkpoint_remote_path': (
                            f'{remote_dir}/model_step_{best["step"]}.pth'
                        ),
                    })

    summary_rows = []
    group_size = len(args.seeds)
    for start in range(0, len(rows), group_size):
        group = rows[start:start + group_size]
        if [row['seed'] for row in group] != args.seeds:
            raise ValueError('Rows are not grouped by the requested seeds')
        first = group[0]
        summary = {
            'dataset': first['dataset'],
            'scenario': first['scenario'],
            'model': first['model'],
            'condition_encoder': first['condition_encoder'],
            'checkpoint': first['checkpoint'],
            'selection_metric': first['selection_metric'],
            'seed_count': group_size,
            'seeds': ';'.join(str(seed) for seed in args.seeds),
            'model_parameters': first['model_parameters'],
            'trainable_parameters': first['trainable_parameters'],
            'best_is_final_count': sum(
                row['best_is_final_step'] for row in group
            ),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in group]
            summary[f'{metric}_mean'] = statistics.mean(values)
            summary[f'{metric}_std'] = statistics.stdev(values)
        summary_rows.append(summary)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / 'runs.csv', RUN_FIELDS, rows)
    write_csv(output_dir / 'summary.csv', SUMMARY_FIELDS, summary_rows)
    print(f'Wrote {len(rows)} runs and {len(summary_rows)} summary rows')


if __name__ == '__main__':
    main()
