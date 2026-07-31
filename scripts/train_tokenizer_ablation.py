"""Train controlled APP, uniform, and learned tokenizer ablations on ShipBench."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.metrics import MetricsCalculator
from src.data_processor.tokenizer_partition import match_partition_budget
from src.datasets.samplers import ConditionBatchSampler
from src.datasets.shipBench import MultiConditionIrregularDataset
from src.datasets.temporal import save_data_protocol
from src.models.irregular.tokenizer_ablation import PointTokenLoss, PointTokenOperator


CHANNELS = ['u', 'v', 'w', 'p_rgh']


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_tokenization_specs(dataset, variant, target_tokens, partition_kwargs):
    specs = []
    for sub_dataset in dataset.sub_datasets:
        coords = sub_dataset._all_coords[0]
        adaptive = match_partition_budget(
            coords, target_tokens=target_tokens, adaptive=True, **partition_kwargs
        )
        if variant == 'adaptive':
            active = adaptive
        elif variant == 'uniform':
            active = match_partition_budget(
                coords,
                target_tokens=adaptive.num_tokens,
                adaptive=False,
                **partition_kwargs,
            )
        elif variant == 'learned':
            active = None
        else:
            raise ValueError(f'Unknown tokenizer variant: {variant}')

        specs.append({
            'condition': sub_dataset.data_dir,
            'num_points': int(sub_dataset.n_points),
            'num_tokens': int(adaptive.num_tokens if active is None else active.num_tokens),
            'reference_adaptive_tokens': int(adaptive.num_tokens),
            'reference_adaptive_capacity': int(adaptive.patch_capacity),
            'patch_capacity': None if active is None else int(active.patch_capacity),
            'token_ids': None if active is None else torch.from_numpy(active.token_ids).long(),
        })
    return specs


def build_region_masks(dataset, partition_kwargs):
    masks = []
    ref_x, ref_y = partition_kwargs['ref_point']
    r1 = partition_kwargs['distance_threshold_1'] * partition_kwargs['ship_length']
    r2 = partition_kwargs['distance_threshold_2'] * partition_kwargs['ship_length']
    for sub_dataset in dataset.sub_datasets:
        coords = sub_dataset._all_coords[0]
        distance = np.sqrt(
            np.square(coords[:, 0] - ref_x) + np.square(coords[:, 1] - ref_y)
        )
        groups = {
            'near': distance < r1,
            'mid': (distance >= r1) & (distance < r2),
            'far': distance >= r2,
        }

        k = min(9, len(coords))
        knn_distance = KDTree(coords).query(coords, k=k)[0][:, -1]
        boundaries = np.quantile(knn_distance, [0.25, 0.5, 0.75])
        groups.update({
            'density_q1_densest': knn_distance <= boundaries[0],
            'density_q2': (knn_distance > boundaries[0]) & (knn_distance <= boundaries[1]),
            'density_q3': (knn_distance > boundaries[1]) & (knn_distance <= boundaries[2]),
            'density_q4_sparsest': knn_distance > boundaries[2],
        })
        masks.append({name: torch.from_numpy(mask) for name, mask in groups.items()})
    return masks


def _condition_id(condition_ids: torch.Tensor) -> int:
    unique = torch.unique(condition_ids)
    if len(unique) != 1:
        raise RuntimeError('Tokenizer ablation requires condition-homogeneous batches')
    return int(unique.item())


def forward_batch(model, batch, specs, device, diagnostics=False):
    positions, flow, target, condition_ids = batch
    condition_id = _condition_id(condition_ids)
    spec = specs[condition_id]
    token_ids = spec['token_ids']
    if token_ids is not None:
        token_ids = token_ids.to(device, non_blocking=True)
    positions = positions.to(device, non_blocking=True)
    flow = flow.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    prediction = model(
        positions,
        flow,
        num_tokens=spec['num_tokens'],
        token_ids=token_ids,
        return_diagnostics=diagnostics,
    )
    return prediction, target, condition_id


def train_one_step(model, batch, specs, criterion, optimizer, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, target, _condition_id_value = forward_batch(
        model, batch, specs, device, diagnostics=False
    )
    loss = criterion(prediction, target)
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.inference_mode()
def validate(model, dataloader, specs, region_masks, criterion, device, disable_tqdm):
    model.eval()
    overall = MetricsCalculator()
    per_condition = [MetricsCalculator() for _ in specs]
    group_calculators = {
        name: MetricsCalculator() for name in region_masks[0]
    }
    total_loss = 0.0
    n_samples = 0
    diagnostic_sums = {}

    for batch in tqdm(dataloader, desc='Validation', disable=disable_tqdm):
        output, target, condition_id = forward_batch(
            model, batch, specs, device, diagnostics=True
        )
        prediction, diagnostics = output
        loss = criterion(prediction, target)
        batch_size = prediction.shape[0]
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        overall.update(prediction, target)
        per_condition[condition_id].update(prediction, target)

        for name, mask in region_masks[condition_id].items():
            expanded = mask.to(device).unsqueeze(0).expand(batch_size, -1)
            group_calculators[name].update(prediction, target, expanded)
        for name, value in diagnostics.items():
            diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + float(value) * batch_size

    return {
        'loss': total_loss / n_samples,
        'overall': overall.compute(),
        'conditions': {
            str(index): calculator.compute()
            for index, calculator in enumerate(per_condition)
        },
        'groups': {
            name: calculator.compute()
            for name, calculator in group_calculators.items()
        },
        'tokenizer_diagnostics': {
            name: value / n_samples for name, value in diagnostic_sums.items()
        },
    }


def profile_forward(model, dataloader, specs, device, warmup, repeats):
    if device.type != 'cuda':
        return None
    batch = next(iter(dataloader))
    positions, flow, _target, condition_ids = batch
    condition_id = _condition_id(condition_ids)
    spec = specs[condition_id]
    positions = positions.to(device)
    flow = flow.to(device)
    token_ids = spec['token_ids']
    if token_ids is not None:
        token_ids = token_ids.to(device)

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(positions, flow, spec['num_tokens'], token_ids)
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(positions, flow, spec['num_tokens'], token_ids)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))

    values = np.asarray(samples, dtype=np.float64)
    return {
        'protocol': 'full PointTokenOperator forward; CUDA Events',
        'condition_id': condition_id,
        'batch_size': int(positions.shape[0]),
        'num_points': int(positions.shape[1]),
        'num_tokens': int(spec['num_tokens']),
        'warmup': warmup,
        'repeats': repeats,
        'median_ms': float(np.median(values)),
        'q25_ms': float(np.quantile(values, 0.25)),
        'q75_ms': float(np.quantile(values, 0.75)),
        'mean_ms': float(values.mean()),
    }


def init_metrics_csv(path):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['step', 'train_loss', 'val_loss', 'mae', 'mse', 'rmse',
                        'relative_l2', 'elapsed_sec'],
        )
        writer.writeheader()


def append_metrics_csv(path, record):
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['step', 'train_loss', 'val_loss', 'mae', 'mse', 'rmse',
                        'relative_l2', 'elapsed_sec'],
            extrasaction='ignore',
        )
        writer.writerow(record)


def main():
    parser = argparse.ArgumentParser(description='Controlled tokenizer ablation')
    parser.add_argument('--variant', required=True,
                        choices=['adaptive', 'uniform', 'learned'])
    parser.add_argument('--data_dirs', nargs='+', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--token_budget', type=int, default=256)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=56)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=16000)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--rollout_holdout_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42, help='Model/training seed')
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ship_length', type=float, default=7.0)
    parser.add_argument('--ref_point', type=float, nargs=2, default=(3.0, 0.0))
    parser.add_argument('--distance_threshold_1', type=float, default=1.0)
    parser.add_argument('--distance_threshold_2', type=float, default=1.5)
    parser.add_argument('--profile_warmup', type=int, default=10)
    parser.add_argument('--profile_repeats', type=int, default=30)
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()

    if args.max_steps < 1 or args.eval_every < 1:
        raise ValueError('max_steps and eval_every must be positive')
    if args.max_steps % args.eval_every != 0:
        raise ValueError('max_steps must be divisible by eval_every')

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / 'metrics.csv'
    metrics_json = output_dir / 'metrics.json'
    init_metrics_csv(metrics_csv)

    train_dataset = MultiConditionIrregularDataset(
        data_dirs=args.data_dirs,
        split='train',
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        rollout_holdout_steps=args.rollout_holdout_steps,
    )
    val_dataset = MultiConditionIrregularDataset.from_existing(
        train_dataset, split='test'
    )
    save_data_protocol(str(output_dir), train_dataset, val_dataset)

    partition_kwargs = {
        'ship_length': args.ship_length,
        'ref_point': tuple(args.ref_point),
        'distance_threshold_1': args.distance_threshold_1,
        'distance_threshold_2': args.distance_threshold_2,
    }
    specs = build_tokenization_specs(
        train_dataset, args.variant, args.token_budget, partition_kwargs
    )
    if max(spec['num_tokens'] for spec in specs) > args.max_tokens:
        raise ValueError('max_tokens is smaller than a resolved partition')
    region_masks = build_region_masks(train_dataset, partition_kwargs)

    tokenization_manifest = {
        'variant': args.variant,
        'requested_token_budget': args.token_budget,
        'max_tokens': args.max_tokens,
        'conditions': [
            {key: value for key, value in spec.items() if key != 'token_ids'}
            for spec in specs
        ],
    }
    with (output_dir / 'tokenization_protocol.json').open('w', encoding='utf-8') as handle:
        json.dump(tokenization_manifest, handle, indent=2)

    set_seed(args.seed)
    model_kwargs = {
        'input_dim': 2 + train_dataset.n_channels,
        'output_dim': train_dataset.n_channels,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'max_tokens': args.max_tokens,
        'tokenizer': 'learned' if args.variant == 'learned' else 'hard',
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointTokenOperator(**model_kwargs).to(device)
    criterion = PointTokenLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        'args': vars(args),
        'model_kwargs': model_kwargs,
        'model_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'device': str(device),
        'channels': CHANNELS,
    }
    with (output_dir / 'run_config.json').open('w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)

    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
    }
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ConditionBatchSampler(train_dataset, args.batch_size, shuffle=True),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=ConditionBatchSampler(val_dataset, args.batch_size, shuffle=False),
        **loader_kwargs,
    )

    print(json.dumps(config, indent=2))
    print(json.dumps(tokenization_manifest, indent=2))
    print(f'Train samples={len(train_dataset)} val samples={len(val_dataset)}')

    disable_tqdm = args.disable_tqdm or not sys.stderr.isatty()
    train_iterator = iter(train_loader)
    losses = []
    records = []
    started = time.time()
    progress = tqdm(total=args.max_steps, desc='Training', disable=disable_tqdm)
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        loss = train_one_step(model, batch, specs, criterion, optimizer, device)
        losses.append(loss)
        progress.update(1)
        if step % args.log_every == 0:
            progress.set_postfix(loss=f'{loss.item():.6f}')
        if step % args.eval_every != 0:
            continue

        train_loss = float(torch.stack(losses).mean().item())
        losses = []
        evaluation = validate(
            model, val_loader, specs, region_masks, criterion, device, disable_tqdm
        )
        overall = evaluation['overall']
        record = {
            'step': step,
            'train_loss': train_loss,
            'val_loss': evaluation['loss'],
            'mae': overall['mae'],
            'mse': overall['mse'],
            'rmse': overall['rmse'],
            'relative_l2': overall['relative_l2'],
            'mae_per_channel': overall['mae_per_channel'],
            'rmse_per_channel': overall['rmse_per_channel'],
            'relative_l2_per_channel': overall['relative_l2_per_channel'],
            'conditions': evaluation['conditions'],
            'groups': evaluation['groups'],
            'tokenizer_diagnostics': evaluation['tokenizer_diagnostics'],
            'elapsed_sec': time.time() - started,
        }
        records.append(record)
        append_metrics_csv(metrics_csv, record)
        print(
            f"step {step:6d} train={train_loss:.6f} val={evaluation['loss']:.6f} "
            f"MAE={overall['mae']:.6f} RMSE={overall['rmse']:.6f}"
        )
    progress.close()

    torch.save(model.state_dict(), output_dir / 'model_final.pth')
    with metrics_json.open('w', encoding='utf-8') as handle:
        json.dump(records, handle, indent=2)
    profile = profile_forward(
        model, val_loader, specs, device,
        args.profile_warmup, args.profile_repeats,
    )
    if profile is not None:
        with (output_dir / 'forward_profile.json').open('w', encoding='utf-8') as handle:
            json.dump(profile, handle, indent=2)
    print(f'Completed tokenizer ablation: {output_dir}')


if __name__ == '__main__':
    main()
