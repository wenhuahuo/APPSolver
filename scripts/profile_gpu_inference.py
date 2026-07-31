"""Measure batch-1 CUDA forward latency on a representative DTC sample."""

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from scripts.evaluate_rollout import (
    APP_METHODS,
    PCNO_K_NEIGHBORS,
    PCNO_N_MODES,
    PATCH_SIZE,
    DOWNSAMPLE_RATIO,
    build_app_model,
    build_irregular_model,
    load_checkpoint,
)
from src.datasets.shipBench import MultiConditionIrregularDataset, MultiConditionPatchDataset
from src.models.irregular.pcno import build_aux_from_pos, collate_aux_batch, compute_fourier_modes


METHODS = [
    'app_transformer', 'app_dpt', 'transolver', 'fno',
    'fusion_deeponet', 'upt', 'gnot', 'pcno',
]


@torch.inference_mode()
def measure_cuda(forward, warmup, repeats):
    for _ in range(warmup):
        forward()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        forward()
        end.record()
    torch.cuda.synchronize()
    return np.asarray(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=np.float64,
    )


def main():
    parser = argparse.ArgumentParser(description='Profile ShipBench model CUDA forward latency')
    parser.add_argument('--run_root', required=True)
    parser.add_argument('--data_dirs', nargs=2, required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for GPU timing')

    device = torch.device('cuda')
    torch.manual_seed(42)

    irregular = MultiConditionIrregularDataset(
        data_dirs=args.data_dirs, split='train', seed=42,
        rollout_holdout_steps=50,
    )
    patch = MultiConditionPatchDataset(
        data_dirs=args.data_dirs, split='train', patch_size=PATCH_SIZE,
        enable_downsample=True, downsample_method='distance',
        downsample_ratio=DOWNSAMPLE_RATIO, seed=42,
        rollout_holdout_steps=50,
    )
    patch_shape = patch.get_global_shape()

    pos, flow, _target = irregular.sub_datasets[0][0]
    pos = pos.unsqueeze(0).to(device)
    flow = flow.unsqueeze(0).to(device)

    patch_sample = patch[0]
    patch_input = patch_sample['input'].unsqueeze(0).to(device)
    patch_mask = patch_sample['mask'].unsqueeze(0).to(device)

    mins = np.full(2, np.inf)
    maxs = np.full(2, -np.inf)
    for sub_dataset in irregular.sub_datasets:
        ref_pos = sub_dataset.coords[0]
        mins = np.minimum(mins, ref_pos.min(axis=0))
        maxs = np.maximum(maxs, ref_pos.max(axis=0))
    fourier_modes = compute_fourier_modes(
        2, [PCNO_N_MODES] * 2, ((maxs - mins) + 1e-6).tolist()
    )
    pcno_aux = {
        key: value.to(device)
        for key, value in collate_aux_batch([
            build_aux_from_pos(
                irregular.sub_datasets[0].coords[0],
                k_neighbors=PCNO_K_NEIGHBORS,
                nmeasures=1,
            )
        ]).items()
    }

    records = []
    for method in METHODS:
        checkpoint = os.path.join(args.run_root, method, 'seed42', 'model_final.pth')
        if method in APP_METHODS:
            model = load_checkpoint(
                build_app_model(method, patch_shape, patch_shape['num_patches']),
                checkpoint,
                device,
            )
            if method == 'app_dpt':
                forward = lambda: model(
                    patch_input, mask=patch_mask, params_embed=None
                )
            else:
                forward = lambda: model(patch_input, params_embed=None)
            input_units = patch_shape['num_patches']
        else:
            model = load_checkpoint(
                build_irregular_model(method, irregular.n_channels, fourier_modes),
                checkpoint,
                device,
            )
            if method == 'pcno':
                forward = lambda: model(
                    pos, flow, pcno_aux['node_weights'],
                    pcno_aux['directed_edges'],
                    pcno_aux['edge_gradient_weights'],
                    pcno_aux['node_mask'],
                )
            else:
                forward = lambda: model(pos, flow)
            input_units = pos.shape[1]

        samples = measure_cuda(forward, args.warmup, args.repeats)
        record = {
            'method': method,
            'input_units': int(input_units),
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'median_ms': float(np.median(samples)),
            'mean_ms': float(np.mean(samples)),
            'std_ms': float(np.std(samples)),
            'q25_ms': float(np.quantile(samples, 0.25)),
            'q75_ms': float(np.quantile(samples, 0.75)),
            'min_ms': float(np.min(samples)),
            'max_ms': float(np.max(samples)),
        }
        records.append(record)
        print(f"{method:20s} median={record['median_ms']:.4f} ms "
              f"IQR=[{record['q25_ms']:.4f}, {record['q75_ms']:.4f}] ms")
        del model, forward
        torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'protocol': 'model forward only; batch size 1; CUDA Events; final checkpoint',
        'sample': args.data_dirs[0],
        'warmup': args.warmup,
        'repeats': args.repeats,
        'gpu': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'python_version': platform.python_version(),
        'records': records,
    }
    with (output_dir / 'gpu_forward_timing.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    with (output_dir / 'gpu_forward_timing.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    main()
