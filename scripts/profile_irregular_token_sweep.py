"""Profile GPU forward latency for completed Transolver/UPT token sweeps."""

import argparse
import csv
import json
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.datasets.shipBench import MultiConditionIrregularDataset
from src.models.irregular.transolver import Transolver
from src.models.irregular.upt import UPT


VARIANT_PATTERN = re.compile(r'^(transolver_slice|upt_tokens)(\d+)$')


def measure_cuda(forward, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            forward()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            forward()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    return np.asarray(samples, dtype=np.float64)


def build_model(model_name, kwargs):
    if model_name == 'transolver':
        return Transolver(**kwargs)
    if model_name == 'upt':
        return UPT(**kwargs)
    raise ValueError(model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_root', type=Path, required=True)
    parser.add_argument('--dataset_root', type=Path, default=Path('datasets/shipBench'))
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')

    device = torch.device('cuda')
    records = []
    for hull in ('DTC', 'KCS', 'KVLCC2'):
        data_dirs = [
            str(args.dataset_root / hull / 'field' / condition)
            for condition in ('1Re', '2Re')
        ]
        dataset = MultiConditionIrregularDataset(
            data_dirs=data_dirs,
            split='train',
            seed=42,
            rollout_holdout_steps=50,
        )
        positions, flow, _target = dataset.sub_datasets[0][0]
        positions = positions.unsqueeze(0).to(device)
        flow = flow.unsqueeze(0).to(device)

        for variant_dir in sorted((args.run_root / hull).iterdir()):
            match = VARIANT_PATTERN.match(variant_dir.name)
            if match is None:
                continue
            model_name = 'transolver' if variant_dir.name.startswith('transolver') else 'upt'
            model_root = variant_dir / model_name
            for seed_dir in sorted(model_root.glob('seed*')):
                config_path = seed_dir / 'run_config.json'
                checkpoint_path = seed_dir / 'model_final.pth'
                if not config_path.exists() or not checkpoint_path.exists():
                    continue
                with config_path.open(encoding='utf-8') as handle:
                    config = json.load(handle)
                model = build_model(
                    model_name, config['resolved_model_kwargs']
                ).to(device)
                state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state)
                model.eval()
                forward = lambda: model(positions, flow)
                samples = measure_cuda(forward, args.warmup, args.repeats)
                records.append({
                    'hull': hull,
                    'variant': variant_dir.name,
                    'model': model_name,
                    'token_value': int(match.group(2)),
                    'seed': int(seed_dir.name.removeprefix('seed')),
                    'num_points': int(positions.shape[1]),
                    'batch_size': 1,
                    'parameters': sum(parameter.numel() for parameter in model.parameters()),
                    'median_ms': float(np.median(samples)),
                    'mean_ms': float(samples.mean()),
                    'std_ms': float(samples.std()),
                    'q25_ms': float(np.quantile(samples, 0.25)),
                    'q75_ms': float(np.quantile(samples, 0.75)),
                })
                print(
                    f"{hull:7s} {variant_dir.name:24s} {seed_dir.name:6s} "
                    f"median={records[-1]['median_ms']:.4f} ms"
                )
                del model, forward
                torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'protocol': 'model forward only; batch size 1; CUDA Events; final checkpoint',
        'warmup': args.warmup,
        'repeats': args.repeats,
        'gpu': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'python_version': platform.python_version(),
        'records': records,
    }
    with (args.output_dir / 'gpu_forward_timing.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    with (args.output_dir / 'gpu_forward_timing.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f'Profiled {len(records)} runs')


if __name__ == '__main__':
    main()
