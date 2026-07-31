"""Summarize controlled tokenizer and irregular-token sweep experiments."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_STEPS = list(range(2000, 16001, 2000))
EXPECTED_SEEDS = {42, 43, 44}


def load_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def validate_records(records, path):
    steps = [record['step'] for record in records]
    if steps != EXPECTED_STEPS:
        raise ValueError(f'Incomplete validation steps in {path}: {steps}')


def summarize_main(root):
    rows = []
    for metrics_path in sorted(root.glob('*/*/seed*/metrics.json')):
        run_dir = metrics_path.parent
        hull, variant, seed_name = run_dir.parts[-3:]
        records = load_json(metrics_path)
        if not records:
            continue
        validate_records(records, metrics_path)
        final = records[-1]
        config = load_json(run_dir / 'run_config.json')
        tokenization = load_json(run_dir / 'tokenization_protocol.json')
        profile_path = run_dir / 'forward_profile.json'
        profile = load_json(profile_path) if profile_path.exists() else {}
        registered = config['model_parameters']
        tokenizer_overhead = config['model_kwargs']['d_model'] ** 2 + 1
        row = {
            'hull': hull,
            'variant': variant,
            'seed': int(seed_name.removeprefix('seed')),
            'step': final['step'],
            'normalized_mae': final['mae'],
            'normalized_rmse': final['rmse'],
            'normalized_relative_l2': final['relative_l2'],
            'registered_parameters': registered,
            'effective_parameters': (
                registered if variant == 'learned'
                else registered - tokenizer_overhead
            ),
            'token_counts': ','.join(
                str(item['num_tokens']) for item in tokenization['conditions']
            ),
            'forward_median_ms': profile.get('median_ms', np.nan),
            'elapsed_sec': final['elapsed_sec'],
        }
        for group, metrics in final['groups'].items():
            row[f'{group}_normalized_mae'] = metrics['mae']
            row[f'{group}_normalized_relative_l2'] = metrics['relative_l2']
        row.update({
            f'diagnostic_{key}': value
            for key, value in final['tokenizer_diagnostics'].items()
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_auxiliary(root):
    rows = []
    profile_path = root / 'gpu_profile' / 'gpu_forward_timing.csv'
    profile_lookup = {}
    if profile_path.exists():
        profile = pd.read_csv(profile_path)
        profile_lookup = {
            (row.hull, row.variant, int(row.seed)): float(row.median_ms)
            for row in profile.itertuples()
        }
    pattern = re.compile(r'^(transolver|upt)_(slice|tokens)(\d+)$')
    for metrics_path in sorted(root.glob('*/*/*/seed*/metrics.json')):
        run_dir = metrics_path.parent
        hull, variant, model, seed_name = run_dir.parts[-4:]
        match = pattern.match(variant)
        if match is None or model != match.group(1):
            continue
        records = load_json(metrics_path)
        if not records:
            continue
        validate_records(records, metrics_path)
        final = records[-1]
        config = load_json(run_dir / 'run_config.json')
        model_rollout_path = run_dir / 'rollout_metrics.json'
        persistence_rollout_path = (
            run_dir.parent.parent / 'persistence' / seed_name / 'rollout_metrics.json'
        )
        row = {
            'hull': hull,
            'variant': variant,
            'model': model,
            'token_value': int(match.group(3)),
            'seed': int(seed_name.removeprefix('seed')),
            'step': final['step'],
            'normalized_mae': final['mae'],
            'normalized_rmse': final['rmse'],
            'normalized_relative_l2': final['relative_l2'],
            'parameters': config['model_parameters'],
            'forward_median_ms': profile_lookup.get(
                (hull, variant, int(seed_name.removeprefix('seed'))), np.nan
            ),
            'elapsed_sec': final['elapsed_sec'],
        }
        if model_rollout_path.exists() and persistence_rollout_path.exists():
            rollout = load_json(model_rollout_path)['overall']
            persistence = load_json(persistence_rollout_path)['overall']
            row['rollout_h50_normalized_mae'] = rollout[-1]['mae']
            row['rollout_h50_error_ratio'] = rollout[-1]['mae'] / persistence[-1]['mae']
            row['rollout_cumulative_error_ratio'] = (
                sum(item['mae'] for item in rollout)
                / sum(item['mae'] for item in persistence)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(frame, keys):
    numeric = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {'seed', 'step'}
    ]
    if frame.empty:
        return frame
    return frame.groupby(keys)[numeric].agg(['mean', 'std']).reset_index()


def flatten_columns(frame):
    flattened = frame.copy()
    flattened.columns = [
        '_'.join(str(part) for part in column if part).rstrip('_')
        if isinstance(column, tuple) else column
        for column in flattened.columns
    ]
    return flattened


def validate_completeness(main_runs, aux_runs):
    if len(main_runs) != 27:
        raise ValueError(f'Expected 27 main runs, found {len(main_runs)}')
    if len(aux_runs) != 81:
        raise ValueError(f'Expected 81 auxiliary runs, found {len(aux_runs)}')
    for keys, group in main_runs.groupby(['hull', 'variant']):
        if set(group['seed']) != EXPECTED_SEEDS:
            raise ValueError(f'Incomplete main seeds for {keys}')
    for keys, group in aux_runs.groupby(['hull', 'model', 'token_value']):
        if set(group['seed']) != EXPECTED_SEEDS:
            raise ValueError(f'Incomplete auxiliary seeds for {keys}')
    if aux_runs['forward_median_ms'].isna().any():
        raise ValueError('Auxiliary GPU profiling is incomplete')

    for hull, group in main_runs.groupby('hull'):
        counts = {
            variant: tuple(int(value) for value in rows.iloc[0]['token_counts'].split(','))
            for variant, rows in group.groupby('variant')
        }
        if counts['adaptive'] != counts['learned']:
            raise ValueError(f'Adaptive/learned token mismatch for {hull}: {counts}')
        for adaptive, uniform in zip(counts['adaptive'], counts['uniform']):
            if abs(adaptive - uniform) / adaptive > 0.02:
                raise ValueError(
                    f'Uniform token budget differs by more than 2% for {hull}: {counts}'
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_root', type=Path, required=True)
    parser.add_argument('--aux_root', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--allow_incomplete', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_runs = summarize_main(args.main_root)
    aux_runs = summarize_auxiliary(args.aux_root)
    if not args.allow_incomplete:
        validate_completeness(main_runs, aux_runs)
    main_runs.to_csv(args.output_dir / 'main_runs.csv', index=False)
    aux_runs.to_csv(args.output_dir / 'auxiliary_runs.csv', index=False)
    flatten_columns(aggregate(main_runs, ['hull', 'variant'])).to_csv(
        args.output_dir / 'main_summary.csv', index=False
    )
    flatten_columns(aggregate(aux_runs, ['hull', 'model', 'token_value'])).to_csv(
        args.output_dir / 'auxiliary_summary.csv', index=False
    )
    print(f'Main runs: {len(main_runs)}; auxiliary runs: {len(aux_runs)}')
    print(f'Wrote summaries to {args.output_dir}')


if __name__ == '__main__':
    main()
