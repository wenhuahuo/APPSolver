"""
Train PCNO (Point Cloud Neural Operator) on Irregular Mesh Flow Field Data.
Supports both ship and CFDBench datasets, and multi-condition training.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.metrics import MetricsCalculator
from src.datasets.cfdBench import (
    CFDBenchIrregularDataset,
    MultiConditionCFDBenchIrregularDataset,
)
from src.datasets.samplers import ConditionBatchSampler
from src.datasets.shipBench import (
    IrregularFlowFieldDataset,
    MultiConditionIrregularDataset,
)
from src.datasets.temporal import save_data_protocol
from src.models.irregular.pcno import (
    PCNO,
    PCNOLoss,
    build_aux_from_pos,
    compute_fourier_modes,
    load_pcno_aux_cache,
)

# ---------------------------------------------------------------------------
# Dataset wrapper: attaches precomputed aux to each sample
# ---------------------------------------------------------------------------

class PCNODataset(Dataset):
    """
    Wraps an irregular dataset and attaches precomputed PCNO auxiliary data
    (node_weights, edges, etc.) to each sample.

    For single-condition datasets, we precompute one aux from coords[0].
    For multi-condition datasets, we precompute one aux per condition.
    """

    def __init__(
        self, base_dataset, k_neighbors: int = 8, nmeasures: int = 1,
        aux_by_condition=None, aux_cache_filename=None,
    ):
        self.base = base_dataset
        if aux_by_condition is not None:
            self.aux_by_condition = aux_by_condition
            return
        self.aux_by_condition = {}

        is_multi = isinstance(
            base_dataset,
            (MultiConditionIrregularDataset, MultiConditionCFDBenchIrregularDataset),
        )
        if aux_cache_filename and not isinstance(
            base_dataset,
            (CFDBenchIrregularDataset, MultiConditionCFDBenchIrregularDataset),
        ):
            raise ValueError('PCNO auxiliary cache is only supported for CFDBench')

        if is_multi:
            action = 'Loading' if aux_cache_filename else 'Precomputing'
            print(f"  {action} PCNO aux data for multi-condition dataset "
                  f"({base_dataset.num_conditions} conditions, {len(base_dataset)} samples)")
            for cond_id in range(base_dataset.num_conditions):
                sub_ds = base_dataset.get_sub_dataset(cond_id)
                ref_pos = sub_ds.coords[0]
                print(f"    condition={cond_id}: N={ref_pos.shape[0]}, k={k_neighbors}")
                if aux_cache_filename:
                    cache_path = os.path.join(
                        sub_ds.root, sub_ds.benchmark, sub_ds.case,
                        aux_cache_filename,
                    )
                    aux = load_pcno_aux_cache(
                        cache_path, ref_pos, k_neighbors, nmeasures,
                    )
                else:
                    aux = build_aux_from_pos(
                        ref_pos,
                        k_neighbors=k_neighbors,
                        nmeasures=nmeasures,
                    )
                self.aux_by_condition[cond_id] = aux
        else:
            ref_pos = base_dataset.coords[0]
            if aux_cache_filename:
                if not isinstance(base_dataset, CFDBenchIrregularDataset):
                    raise ValueError('PCNO auxiliary cache is only supported for CFDBench')
                cache_path = os.path.join(
                    base_dataset.root, base_dataset.benchmark, base_dataset.case,
                    aux_cache_filename,
                )
                self.aux_by_condition[0] = load_pcno_aux_cache(
                    cache_path, ref_pos, k_neighbors, nmeasures,
                )
            else:
                print(f"  Precomputing PCNO aux data for {len(base_dataset)} samples "
                      f"(N={ref_pos.shape[0]}, k={k_neighbors}) ...")
                self.aux_by_condition[0] = build_aux_from_pos(
                    ref_pos,
                    k_neighbors=k_neighbors,
                    nmeasures=nmeasures,
                )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        if len(sample) == 4:
            pos, fx, y, cond_id = sample
            cond_id_int = int(cond_id.item()) if torch.is_tensor(cond_id) else int(cond_id)
        else:
            pos, fx, y = sample
            cond_id_int = 0

        return pos, fx, y, cond_id_int


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pcno_collate_fn(batch):
    """Stack one same-condition batch and retain its geometry identifier."""
    pos_list, fx_list, y_list, condition_ids = zip(*batch, strict=True)
    return (
        torch.stack(pos_list),
        torch.stack(fx_list),
        torch.stack(y_list),
        int(condition_ids[0]),
    )


def move_aux_to_device(aux_by_condition, device):
    return {
        condition_id: {
            key: torch.from_numpy(value).to(device)
            for key, value in aux.items()
        }
        for condition_id, aux in aux_by_condition.items()
    }


def expand_aux(aux, batch_size):
    return {
        key: value.unsqueeze(0).expand(batch_size, *value.shape)
        for key, value in aux.items()
    }


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


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(
    model, dataloader, criterion, optimizer, optimizer_inv_L, device,
    aux_by_condition, disable_tqdm=False,
):
    model.train()
    total_loss = 0
    n_samples  = 0

    for pos, fx, target, condition_id in tqdm(dataloader, desc="Training", disable=disable_tqdm):
        pos = pos.to(device, non_blocking=True)
        fx = fx.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        aux = expand_aux(aux_by_condition[condition_id], pos.size(0))
        node_weights = aux['node_weights']
        directed_edges = aux['directed_edges']
        edge_gradient_weights = aux['edge_gradient_weights']
        node_mask = aux['node_mask']

        optimizer.zero_grad(set_to_none=True)
        if optimizer_inv_L:
            optimizer_inv_L.zero_grad(set_to_none=True)

        pred = model(pos, fx, node_weights, directed_edges,
                     edge_gradient_weights, node_mask)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        if optimizer_inv_L:
            optimizer_inv_L.step()

        total_loss += loss.item() * pos.size(0)
        n_samples  += pos.size(0)

    return total_loss / n_samples


def train_one_step(
    model, batch, criterion, optimizer, optimizer_inv_L, device,
    aux_by_condition,
):
    model.train()
    pos, fx, target, condition_id = batch
    pos = pos.to(device, non_blocking=True)
    fx = fx.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    aux = expand_aux(aux_by_condition[condition_id], pos.size(0))
    node_weights = aux['node_weights']
    directed_edges = aux['directed_edges']
    edge_gradient_weights = aux['edge_gradient_weights']
    node_mask = aux['node_mask']

    optimizer.zero_grad(set_to_none=True)
    if optimizer_inv_L:
        optimizer_inv_L.zero_grad(set_to_none=True)

    pred = model(pos, fx, node_weights, directed_edges,
                 edge_gradient_weights, node_mask)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    if optimizer_inv_L:
        optimizer_inv_L.step()
    return loss.detach()


def validate(
    model, dataloader, criterion, device, aux_by_condition, disable_tqdm=False
):
    model.eval()
    total_loss   = 0
    n_samples    = 0
    metrics_calc = MetricsCalculator()

    with torch.no_grad():
        for pos, fx, target, condition_id in tqdm(dataloader, desc="Validation", disable=disable_tqdm):
            pos = pos.to(device, non_blocking=True)
            fx = fx.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            aux = expand_aux(aux_by_condition[condition_id], pos.size(0))
            node_weights = aux['node_weights']
            directed_edges = aux['directed_edges']
            edge_gradient_weights = aux['edge_gradient_weights']
            node_mask = aux['node_mask']

            pred = model(pos, fx, node_weights, directed_edges,
                         edge_gradient_weights, node_mask)

            loss = criterion(pred, target)

            total_loss += loss.item() * pos.size(0)
            n_samples  += pos.size(0)
            metrics_calc.update(pred, target)

    metrics = metrics_calc.compute()
    return total_loss / n_samples, metrics


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _parse_cfd_dirs(data_dirs):
    parts_list = [d.split('/') for d in data_dirs]
    roots = ['/'.join(p[:-2]) if len(p) >= 2 else d for p, d in zip(parts_list, data_dirs, strict=True)]
    benchmarks = [p[-2] if len(p) >= 2 else '03_damflow' for p in parts_list]
    cases = [p[-1] if len(p) >= 1 else 'case0' for p in parts_list]
    return roots, benchmarks, cases


def _make_single_dataset(
    dataset_type, data_dir, step_size, train_ratio, seed, split,
    rollout_holdout_steps=0, normalization_params=None,
):
    common = {
        'step_size': step_size, 'train_ratio': train_ratio, 'seed': seed,
        'normalize': True, 'rollout_holdout_steps': rollout_holdout_steps,
        'normalization_params': normalization_params,
    }
    if dataset_type == 'ship':
        ds = IrregularFlowFieldDataset(data_dir=data_dir, **common)
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
    common = {
        'step_size': step_size, 'train_ratio': train_ratio, 'seed': seed,
        'normalize': True, 'split': split,
        'rollout_holdout_steps': rollout_holdout_steps,
        'normalization_params': normalization_params,
    }
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
    rollout_holdout_steps=0,
):
    holdout = rollout_holdout_steps if dataset_type == 'ship' else 0
    if not multi_condition:
        train_ds = _make_single_dataset(
            dataset_type, data_dirs[0], step_size, train_ratio, seed, 'train',
            rollout_holdout_steps=holdout,
        )
        if dataset_type == 'ship':
            val_ds = train_ds.clone_for_split('test')
        else:
            val_ds = _make_single_dataset(
                dataset_type, data_dirs[0], step_size, train_ratio, seed, 'test',
                rollout_holdout_steps=holdout,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train PCNO')
    parser.add_argument('--dataset_type', type=str, default='ship',
                        choices=['ship', 'cfd_bench'],
                        help='Dataset type: ship or cfd_bench')
    parser.add_argument('--data_dirs', type=str, nargs='+',
                        default=['datasets/shipBench/DTC/field/1Re'],
                        help='Training data path(s). Multiple paths require --multi_condition')
    parser.add_argument('--multi_condition', action='store_true', default=False,
                        help='Enable multi-condition training mode')
    parser.add_argument('--data_dir', type=str,
                        default=None,
                        help='Deprecated: single data path. Prefer --data_dirs')
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--max_steps',   type=int,   default=0,
                        help='If >0, use step-based training and ignore --epochs')
    parser.add_argument('--eval_every',  type=int,   default=200,
                        help='Validation interval in step-based mode')
    parser.add_argument('--log_every',   type=int,   default=50,
                        help='Logging interval in step-based mode')
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--layers',      type=int,   nargs='+',
                        default=[64, 64, 64, 64],
                        help='Hidden channel widths per PCNO layer')
    parser.add_argument('--fc_dim',      type=int,   default=0,
                        help='Output projection MLP hidden dim (0 = linear)')
    parser.add_argument('--n_modes',     type=int,   default=4,
                        help='Number of Fourier modes per dimension')
    parser.add_argument('--nmeasures',   type=int,   default=1,
                        help='Number of integration measures')
    parser.add_argument('--k_neighbors', type=int,   default=8,
                        help='k-NN neighbours for graph construction')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--rollout_holdout_steps', type=int, default=50,
                        help='Reserved contiguous ShipBench rollout horizon (0 disables)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Dataset split seed (default: --seed)')
    parser.add_argument('--aux_cache_filename', type=str, default=None,
                        help='Required per-case PCNO auxiliary cache for CFDBench')
    parser.add_argument('--disable_tqdm', action='store_true', default=False,
                        help='Disable tqdm progress bars (recommended for log redirection)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker processes')
    parser.add_argument('--save_dir',    type=str,
                        default='outputs_pcno', help='Output directory')
    args = parser.parse_args()

    if args.data_dir is not None and args.data_dirs == ['datasets/shipBench/DTC/field/1Re']:
        args.data_dirs = [args.data_dir]
    if len(args.data_dirs) > 1 and not args.multi_condition:
        raise ValueError('Multiple --data_dirs require --multi_condition')
    split_seed = args.seed if args.split_seed is None else args.split_seed

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Multi-condition: {args.multi_condition}")
    print(f"Data dirs: {args.data_dirs}")
    print(f"Model seed: {args.seed}; split seed: {split_seed}")
    print(f"Rollout holdout: {args.rollout_holdout_steps if args.dataset_type == 'ship' else 0} steps")

    disable_tqdm = bool(args.disable_tqdm or (not sys.stderr.isatty()))
    print(f"Disable tqdm: {disable_tqdm}")

    os.makedirs(args.save_dir, exist_ok=True)
    metrics_csv_path = os.path.join(args.save_dir, 'metrics.csv')
    metrics_json_path = os.path.join(args.save_dir, 'metrics.json')
    _init_metrics_csv(metrics_csv_path)
    metrics_records = []
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load base datasets
    # ------------------------------------------------------------------
    print("Loading datasets...")
    train_base, val_base = create_datasets(
        args.dataset_type,
        args.data_dirs,
        1,
        args.train_ratio,
        split_seed,
        args.multi_condition,
        rollout_holdout_steps=args.rollout_holdout_steps,
    )
    save_data_protocol(args.save_dir, train_base, val_base)

    if args.multi_condition:
        print(f"Train dataset: {len(train_base)} samples "
              f"(multi-condition, {train_base.num_conditions} conditions)")
    else:
        print(f"Train dataset: {len(train_base)} samples")
        print(f"Coords shape:  {train_base.coords.shape}")
        print(f"Flows shape:   {train_base.flows.shape}")
    print(f"Val dataset:   {len(val_base)} samples")

    # ------------------------------------------------------------------
    # Build Fourier modes from domain extent
    # ------------------------------------------------------------------
    if args.multi_condition:
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        for cond_id in range(train_base.num_conditions):
            sub_ds = train_base.get_sub_dataset(cond_id)
            pos = sub_ds.coords[0]
            min_x = min(min_x, float(pos[..., 0].min()))
            max_x = max(max_x, float(pos[..., 0].max()))
            min_y = min(min_y, float(pos[..., 1].min()))
            max_y = max(max_y, float(pos[..., 1].max()))
        Lx = (max_x - min_x) + 1e-6
        Ly = (max_y - min_y) + 1e-6
    else:
        all_coords = train_base.coords
        Lx = float(all_coords[..., 0].max() - all_coords[..., 0].min()) + 1e-6
        Ly = float(all_coords[..., 1].max() - all_coords[..., 1].min()) + 1e-6
    nks = [args.n_modes, args.n_modes] * args.nmeasures
    Ls  = [Lx, Ly] * args.nmeasures
    modes = compute_fourier_modes(2, nks, Ls)
    print(f"Fourier modes: {modes.shape}  (domain Lx={Lx:.3f}, Ly={Ly:.3f})")

    # ------------------------------------------------------------------
    # Wrap with PCNO aux
    # ------------------------------------------------------------------
    print("Building PCNO aux for train split ...")
    train_dataset = PCNODataset(
        train_base,
        k_neighbors=args.k_neighbors,
        nmeasures=args.nmeasures,
        aux_cache_filename=args.aux_cache_filename,
    )
    print("Reusing PCNO aux for val split ...")
    val_dataset = PCNODataset(
        val_base, aux_by_condition=train_dataset.aux_by_condition
    )
    aux_by_condition = move_aux_to_device(
        train_dataset.aux_by_condition, device
    )

    loader_kwargs = {
        'num_workers': args.num_workers,
        'collate_fn': pcno_collate_fn,
        'pin_memory': device.type == 'cuda',
    }
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    if args.multi_condition:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=ConditionBatchSampler(train_base, args.batch_size, shuffle=True),
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=ConditionBatchSampler(val_base, args.batch_size, shuffle=False),
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

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if args.multi_condition:
        in_channels = train_base.n_channels
    else:
        in_channels = train_base.flows.shape[2]
    out_channels = in_channels

    model = PCNO(
        in_channels=in_channels,
        out_channels=out_channels,
        modes=modes,
        layers=args.layers,
        fc_dim=args.fc_dim,
        nmeasures=args.nmeasures,
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {model_params:,} parameters")
    run_config = vars(args).copy()
    run_config.update(
        model_seed=args.seed,
        resolved_split_seed=split_seed,
        model_params=model_params,
    )
    with open(os.path.join(args.save_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)

    criterion = PCNOLoss()
    optimizer = torch.optim.Adam(model.normal_params, lr=args.lr)
    if model.inv_L_params:
        optimizer_inv_L = torch.optim.Adam(model.inv_L_params, lr=args.lr * 0.1)
    else:
        optimizer_inv_L = None

    if args.max_steps > 0:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            div_factor=2,
            final_div_factor=100,
            pct_start=0.2,
            total_steps=args.max_steps,
        )
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            div_factor=2,
            final_div_factor=100,
            pct_start=0.2,
            steps_per_epoch=1,
            epochs=args.epochs,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
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
                model, batch, criterion, optimizer, optimizer_inv_L, device,
                aux_by_condition,
            )
            scheduler.step()
            train_losses_since_eval.append(loss_tensor)

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss_tensor.item():.6f}")

            if step % args.eval_every != 0 and step != args.max_steps:
                continue

            train_loss = float(torch.stack(train_losses_since_eval).mean().item())
            train_losses_since_eval = []
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, aux_by_condition,
                disable_tqdm=disable_tqdm,
            )
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

            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, optimizer_inv_L,
                device, aux_by_condition, disable_tqdm=disable_tqdm,
            )
            print(f"Train Loss: {train_loss:.6f}")

            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, aux_by_condition,
                disable_tqdm=disable_tqdm,
            )
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
            print(f"Val Loss: {val_loss:.6f}, "
                  f"MAE: {val_metrics['mae']:.6f}, "
                  f"MSE: {val_metrics['mse']:.6f}, "
                  f"RMSE: {val_metrics['rmse']:.6f}")

            scheduler.step()

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
