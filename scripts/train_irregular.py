"""
Unified training script for irregular mesh models (Transolver, FNO, FusionDeepONet, UPT, GNOT)
Supports both ship and CFDBench datasets, and multi-condition training
"""

import os
import sys
import argparse
import csv
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.datasets.shipBench import IrregularFlowFieldDataset, MultiConditionIrregularDataset
from src.datasets.cfdBench import CFDBenchIrregularDataset, MultiConditionCFDBenchIrregularDataset
from src.models.irregular import Transolver, TransolverLoss
from src.models.irregular.fno import FNO, FNOLoss
from src.models.irregular.fusion_deeponet import FusionDeepONet, FusionDeepONetLoss
from src.models.irregular.upt import UPT, UPTLoss
from src.models.irregular.gnot import GNOT, GNOTLoss
from src.core.metrics import MetricsCalculator
from src.datasets.samplers import ConditionBatchSampler
from src.datasets.temporal import save_data_protocol


MODEL_REGISTRY = {
    'transolver': (Transolver, TransolverLoss, {
        'space_dim': 2,
        'unified_pos': False,
    }),
    'fno': (FNO, FNOLoss, {
        'space_dim': 2,
        'geotype': 'unstructured',
        'shapelist': [32, 32],
    }),
    'fusion_deeponet': (FusionDeepONet, FusionDeepONetLoss, {
        'coord_dim': 2,
    }),
    'upt': (UPT, UPTLoss, {
        'space_dim': 2,
    }),
    'gnot': (GNOT, GNOTLoss, {
        'space_dim': 2,
    }),
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _init_metrics_csv(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'mode', 'epoch', 'step', 'train_loss',
                'val_loss', 'mae', 'mse', 'rmse', 'relative_l2', 'elapsed_sec'
            ],
            extrasaction='ignore',
        )
        writer.writeheader()


def _append_metrics_csv(csv_path, row):
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'mode', 'epoch', 'step', 'train_loss',
                'val_loss', 'mae', 'mse', 'rmse', 'relative_l2', 'elapsed_sec'
            ],
            extrasaction='ignore',
        )
        writer.writerow(row)


def _unpack_batch(batch, multi_condition):
    """统一解包 batch，兼容有无 condition_id 的情况。"""
    if multi_condition:
        pos, flow, target, _cond = batch
    else:
        pos, flow, target = batch
    return pos, flow, target


def train_epoch(model, dataloader, criterion, optimizer, device, multi_condition=False, disable_tqdm=False):
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in tqdm(dataloader, desc="Training", disable=disable_tqdm):
        pos, flow, target = _unpack_batch(batch, multi_condition)
        pos = pos.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(pos, flow)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pos.size(0)
        n_samples += pos.size(0)

    return total_loss / n_samples


def train_one_step(model, batch, criterion, optimizer, device, multi_condition=False):
    model.train()
    pos, flow, target = _unpack_batch(batch, multi_condition)
    pos = pos.to(device, non_blocking=True)
    flow = flow.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    pred = model(pos, flow)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    return loss.detach()


def validate(model, dataloader, criterion, device, multi_condition=False, disable_tqdm=False):
    model.eval()
    total_loss = 0
    n_samples = 0
    metrics_calc = MetricsCalculator()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", disable=disable_tqdm):
            pos, flow, target = _unpack_batch(batch, multi_condition)
            pos = pos.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(pos, flow)
            loss = criterion(pred, target)

            total_loss += loss.item() * pos.size(0)
            n_samples += pos.size(0)
            metrics_calc.update(pred, target)

    metrics = metrics_calc.compute()
    return total_loss / n_samples, metrics


def _parse_cfd_dirs(data_dirs):
    """将 CFDBench 路径列表解析为 (roots, benchmarks, cases)。"""
    parts_list = [d.split('/') for d in data_dirs]
    roots      = ['/'.join(p[:-2]) if len(p) >= 2 else d for p, d in zip(parts_list, data_dirs)]
    benchmarks = [p[-2] if len(p) >= 2 else '03_damflow' for p in parts_list]
    cases      = [p[-1] if len(p) >= 1 else 'case0'      for p in parts_list]
    return roots, benchmarks, cases


def _make_single_dataset(
    dataset_type, data_dir, step_size, train_ratio, seed, split,
    prefer_cache=True, rollout_holdout_steps=0, normalization_params=None,
):
    common = dict(
        step_size=step_size, train_ratio=train_ratio, seed=seed, normalize=True,
        rollout_holdout_steps=rollout_holdout_steps,
        normalization_params=normalization_params,
    )
    if dataset_type == 'ship':
        ds = IrregularFlowFieldDataset(
            data_dir=data_dir, prefer_cache=prefer_cache, **common,
        )
    elif dataset_type == 'cfd_bench':
        roots, benchmarks, cases = _parse_cfd_dirs([data_dir])
        ds = CFDBenchIrregularDataset(
            root=roots[0], benchmark=benchmarks[0], case=cases[0], **common,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    ds.set_split(split)
    return ds


def _make_multi_dataset(
    dataset_type, data_dirs, step_size, train_ratio, seed, split,
    rollout_holdout_steps=0, normalization_params=None,
):
    common = dict(
        step_size=step_size, train_ratio=train_ratio, seed=seed,
        normalize=True, split=split,
        rollout_holdout_steps=rollout_holdout_steps,
        normalization_params=normalization_params,
    )
    if dataset_type == 'ship':
        return MultiConditionIrregularDataset(data_dirs=data_dirs, **common)
    if dataset_type == 'cfd_bench':
        roots, benchmarks, cases = _parse_cfd_dirs(data_dirs)
        return MultiConditionCFDBenchIrregularDataset(
            roots=roots, benchmarks=benchmarks, cases=cases, **common,
        )
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_datasets(
    dataset_type, data_dirs, step_size, train_ratio, seed, multi_condition,
    prefer_cache=True, rollout_holdout_steps=0,
):
    holdout = rollout_holdout_steps if dataset_type == 'ship' else 0
    if not multi_condition:
        train_ds = _make_single_dataset(
            dataset_type, data_dirs[0], step_size, train_ratio, seed, 'train',
            prefer_cache=prefer_cache, rollout_holdout_steps=holdout,
        )
        if dataset_type == 'ship':
            val_ds = train_ds.clone_for_split('test')
        else:
            val_ds = _make_single_dataset(
                dataset_type, data_dirs[0], step_size, train_ratio, seed, 'test',
                prefer_cache=prefer_cache, rollout_holdout_steps=holdout,
                normalization_params=train_ds.get_normalization_params(),
            )
        return train_ds, val_ds

    train_ds = _make_multi_dataset(
        dataset_type, data_dirs, step_size, train_ratio, seed, 'train',
        rollout_holdout_steps=holdout,
    )
    if dataset_type == 'cfd_bench':
        val_ds = MultiConditionCFDBenchIrregularDataset.from_existing(
            train_ds, split='test', max_points=train_ds.global_max_points,
        )
    else:
        val_ds = MultiConditionIrregularDataset.from_existing(
            train_ds, split='test'
        )
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description='Train Irregular Mesh Models')
    parser.add_argument('--model', type=str, required=True,
                        choices=['transolver', 'fno', 'fusion_deeponet', 'upt', 'gnot'],
                        help='Model type')
    parser.add_argument('--dataset_type', type=str, default='ship', choices=['ship', 'cfd_bench'],
                        help='Dataset type: ship or cfd_bench')
    parser.add_argument('--data_dirs', type=str, nargs='+',
                        default=['datasets/shipBench/DTC/field/1Re'],
                        help='Training data path(s). Single path = single condition, '
                             'multiple paths = multi-condition (requires --multi_condition)')
    parser.add_argument('--multi_condition', action='store_true', default=False,
                        help='Enable multi-condition training mode')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=0,
                        help='If >0, use step-based training and ignore --epochs')
    parser.add_argument('--eval_every', type=int, default=200,
                        help='Validation interval in step-based mode')
    parser.add_argument('--log_every', type=int, default=50,
                        help='Logging interval in step-based mode')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train/val split ratio over valid temporal pairs')
    parser.add_argument('--rollout_holdout_steps', type=int, default=50,
                        help='Reserved contiguous ShipBench rollout horizon (0 disables)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Model initialization and training seed')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Dataset split/rollout seed (default: --seed)')
    parser.add_argument('--disable_tqdm', action='store_true', default=False,
                        help='Disable tqdm progress bars (recommended for log redirection)')
    parser.add_argument('--disable_cache', action='store_true', default=False,
                        help='Load CSV timesteps directly instead of flow_cache.npz')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker processes')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory (default: outputs_<model>)')

    model_parser = parser.add_argument_group('model-specific arguments')
    model_parser.add_argument('--n_layers', type=int)
    model_parser.add_argument('--n_hidden', type=int)
    model_parser.add_argument('--n_heads', type=int)
    model_parser.add_argument('--mlp_ratio', type=int)
    model_parser.add_argument('--modes', type=int, help='Fourier modes (FNO)')
    model_parser.add_argument('--hidden_dim', type=int, help='Hidden dim (FusionDeepONet)')
    model_parser.add_argument('--G_dim', type=int, help='Embedding dim (FusionDeepONet)')
    model_parser.add_argument('--slice_num', type=int, default=32, help='Slice num (Transolver)')
    model_parser.add_argument('--ref', type=int, default=8, help='Ref grid size (Transolver)')
    model_parser.add_argument('--num_output_tokens', type=int, default=None, help='Output tokens (UPT)')
    model_parser.add_argument('--n_experts', type=int, default=3, help='Experts (GNOT)')

    args = parser.parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.slice_num < 1:
        raise ValueError('--slice_num must be positive')
    if args.num_output_tokens is not None and args.num_output_tokens < 1:
        raise ValueError('--num_output_tokens must be positive')

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Multi-condition: {args.multi_condition}")
    print(f"Data dirs: {args.data_dirs}")
    print(f"Model seed: {args.seed}; split seed: {split_seed}")
    print(f"Val split: test split from data_dirs (ratio={args.train_ratio})")
    print(f"Prefer cache: {not args.disable_cache}")
    print(f"Rollout holdout: {args.rollout_holdout_steps if args.dataset_type == 'ship' else 0} steps")

    disable_tqdm = bool(args.disable_tqdm or (not sys.stderr.isatty()))
    print(f"Disable tqdm: {disable_tqdm}")

    if args.save_dir is None:
        args.save_dir = f'outputs_{args.model}'
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_csv_path = os.path.join(args.save_dir, 'metrics.csv')
    metrics_json_path = os.path.join(args.save_dir, 'metrics.json')
    _init_metrics_csv(metrics_csv_path)
    metrics_records = []
    t0 = time.time()

    model_cls, loss_cls, base_kwargs = MODEL_REGISTRY[args.model]

    print(f"\nLoading datasets...")
    train_dataset, val_dataset = create_datasets(
        args.dataset_type, args.data_dirs,
        step_size=1, train_ratio=args.train_ratio, seed=split_seed,
        multi_condition=args.multi_condition,
        prefer_cache=not args.disable_cache,
        rollout_holdout_steps=args.rollout_holdout_steps,
    )
    save_data_protocol(args.save_dir, train_dataset, val_dataset)

    # 判断是否为多工况（含 condition_id 输出）
    val_is_multi = isinstance(val_dataset, (MultiConditionIrregularDataset,
                                            MultiConditionCFDBenchIrregularDataset))

    if args.multi_condition:
        print(f"Train: {len(train_dataset)} samples (multi-condition, "
              f"{train_dataset.num_conditions} conditions)")
        fun_dim = train_dataset.n_channels
    else:
        print(f"Train: {len(train_dataset)} samples")
        fun_dim = train_dataset.flows.shape[2]
    print(f"Val:   {len(val_dataset)} samples")

    out_dim = fun_dim

    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
    }
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    if args.multi_condition:
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
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            **loader_kwargs,
        )

    model_kwargs = {**base_kwargs}
    if args.model in ['transolver', 'fno', 'upt', 'gnot']:
        model_kwargs.update({'fun_dim': fun_dim, 'out_dim': out_dim})
    if args.model == 'transolver':
        model_kwargs.update({
            'n_layers': args.n_layers or 5,
            'n_hidden': args.n_hidden or 160,
            'dropout': 0.0,
            'n_head': args.n_heads or 4,
            'act': 'gelu',
            'mlp_ratio': args.mlp_ratio or 2,
            'slice_num': args.slice_num,
            'ref': args.ref,
        })
    elif args.model == 'fno':
        model_kwargs.update({
            'n_hidden': args.n_hidden or 32,
            'n_layers': args.n_layers or 5,
            'modes': args.modes or 8,
        })
    elif args.model == 'fusion_deeponet':
        model_kwargs.update({
            'in_channels': fun_dim,
            'out_channels': out_dim,
            'hidden_dim': args.hidden_dim or 288,
            'n_layers': args.n_layers or 5,
            'G_dim': args.G_dim or 112,
        })
    elif args.model == 'upt':
        model_kwargs.update({
            'n_hidden': args.n_hidden or 128,
            'n_heads': args.n_heads or 4,
            'n_layers': args.n_layers or 2,
            'num_output_tokens': 64 if args.num_output_tokens is None else args.num_output_tokens,
        })
    elif args.model == 'gnot':
        model_kwargs.update({
            'n_hidden': args.n_hidden or 112,
            'n_heads': args.n_heads or 4,
            'n_layers': args.n_layers or 2,
            'mlp_ratio': args.mlp_ratio or 2,
            'n_experts': args.n_experts,
        })

    model = model_cls(**model_kwargs).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = loss_cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_config = {
        'args': vars(args),
        'model_seed': args.seed,
        'split_seed': split_seed,
        'resolved_model_kwargs': model_kwargs,
        'model_parameters': sum(p.numel() for p in model.parameters()),
    }
    with open(os.path.join(args.save_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)

    if args.max_steps > 0:
        print(f"\nStep-based training: max_steps={args.max_steps}, eval_every={args.eval_every}")
        step = 0
        train_iter = iter(train_loader)
        train_losses_since_eval = []
        pbar = tqdm(total=args.max_steps, desc='Training (steps)', disable=disable_tqdm)

        while step < args.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            step += 1
            pbar.update(1)
            loss_tensor = train_one_step(
                model, batch, criterion, optimizer, device,
                multi_condition=args.multi_condition,
            )
            train_losses_since_eval.append(loss_tensor)

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss_tensor.item():.6f}")

            if step % args.eval_every != 0 and step != args.max_steps:
                continue

            train_loss = float(torch.stack(train_losses_since_eval).mean().item())
            train_losses_since_eval = []
            val_loss, val_metrics = validate(model, val_loader, criterion, device,
                                             multi_condition=val_is_multi,
                                             disable_tqdm=disable_tqdm)
            record = {
                'mode': 'step',
                'epoch': -1,
                'step': step,
                'train_loss': train_loss,
                'val_loss': float(val_loss),
                'mae': float(val_metrics['mae']),
                'mse': float(val_metrics['mse']),
                'rmse': float(val_metrics['rmse']),
                'relative_l2': float(val_metrics['relative_l2']),
                'mae_per_channel': val_metrics['mae_per_channel'],
                'rmse_per_channel': val_metrics['rmse_per_channel'],
                'relative_l2_per_channel': val_metrics['relative_l2_per_channel'],
                'elapsed_sec': float(time.time() - t0),
            }
            metrics_records.append(record)
            _append_metrics_csv(metrics_csv_path, record)
            print(f"step {step:>6d}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}  "
                  f"MAE: {val_metrics['mae']:.6f}  MSE: {val_metrics['mse']:.6f}  RMSE: {val_metrics['rmse']:.6f}")

            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'model_step_{step}.pth'))

        pbar.close()
    else:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                     multi_condition=args.multi_condition,
                                     disable_tqdm=disable_tqdm)
            print(f"Train Loss: {train_loss:.6f}")

            val_loss, val_metrics = validate(model, val_loader, criterion, device,
                                             multi_condition=val_is_multi,
                                             disable_tqdm=disable_tqdm)
            record = {
                'mode': 'epoch',
                'epoch': epoch + 1,
                'step': -1,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'mae': float(val_metrics['mae']),
                'mse': float(val_metrics['mse']),
                'rmse': float(val_metrics['rmse']),
                'relative_l2': float(val_metrics['relative_l2']),
                'mae_per_channel': val_metrics['mae_per_channel'],
                'rmse_per_channel': val_metrics['rmse_per_channel'],
                'relative_l2_per_channel': val_metrics['relative_l2_per_channel'],
                'elapsed_sec': float(time.time() - t0),
            }
            metrics_records.append(record)
            _append_metrics_csv(metrics_csv_path, record)
            print(f"Val Loss: {val_loss:.6f}  MAE: {val_metrics['mae']:.6f}  "
                  f"MSE: {val_metrics['mse']:.6f}  RMSE: {val_metrics['rmse']:.6f}")

            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_final.pth'))
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_records, f, indent=2)
    print(f"\nTraining complete. Model saved to {args.save_dir}/model_final.pth")
    print(f"Validation metrics saved to {metrics_csv_path}")


if __name__ == '__main__':
    main()
