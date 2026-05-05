"""
Train PCNO (Point Cloud Neural Operator) on Irregular Mesh Flow Field Data.
Supports both ship and CFDBench datasets, and multi-condition training.
"""

import os
import sys
import argparse
import csv
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.datasets.shipBench import IrregularFlowFieldDataset, MultiConditionIrregularDataset
from src.datasets.cfdBench import CFDBenchIrregularDataset, MultiConditionCFDBenchIrregularDataset
from src.models.irregular.pcno import (
    PCNO, PCNOLoss,
    compute_fourier_modes, build_aux_from_pos, collate_aux_batch,
)
from src.core.metrics import MetricsCalculator


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

    def __init__(self, base_dataset, k_neighbors: int = 8, nmeasures: int = 1):
        self.base = base_dataset
        self.aux_by_condition = {}

        is_multi = isinstance(
            base_dataset,
            (MultiConditionIrregularDataset, MultiConditionCFDBenchIrregularDataset),
        )

        if is_multi:
            print(f"  Precomputing PCNO aux data for multi-condition dataset "
                  f"({base_dataset.num_conditions} conditions, {len(base_dataset)} samples)")
            for cond_id in range(base_dataset.num_conditions):
                sub_ds = base_dataset.get_sub_dataset(cond_id)
                ref_pos = sub_ds.coords[0]
                if ref_pos.shape[0] > base_dataset.global_max_points:
                    ref_pos = ref_pos[:base_dataset.global_max_points]

                print(f"    condition={cond_id}: N={ref_pos.shape[0]}, k={k_neighbors}")
                self.aux_by_condition[cond_id] = build_aux_from_pos(
                    ref_pos,
                    k_neighbors=k_neighbors,
                    nmeasures=nmeasures,
                )
        else:
            ref_pos = base_dataset.coords[0]
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

        aux = self.aux_by_condition[cond_id_int]
        return pos, fx, y, aux


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pcno_collate_fn(batch):
    """Custom collate: stacks (pos, fx, y) normally; pads edge arrays."""
    pos_list, fx_list, y_list, aux_list = zip(*batch)

    pos = torch.stack(pos_list)   # (B, N, 2)
    fx  = torch.stack(fx_list)    # (B, N, C)
    y   = torch.stack(y_list)     # (B, N, C)

    aux_batch = collate_aux_batch(list(aux_list))
    return pos, fx, y, aux_batch


def _init_metrics_csv(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'mode', 'epoch', 'step', 'train_loss',
                'val_loss', 'mae', 'mse', 'rmse', 'elapsed_sec'
            ]
        )
        writer.writeheader()


def _append_metrics_csv(csv_path, row):
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'mode', 'epoch', 'step', 'train_loss',
                'val_loss', 'mae', 'mse', 'rmse', 'elapsed_sec'
            ]
        )
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device, disable_tqdm=False):
    model.train()
    total_loss = 0
    n_samples  = 0

    for pos, fx, target, aux in tqdm(dataloader, desc="Training", disable=disable_tqdm):
        pos    = pos.to(device)
        fx     = fx.to(device)
        target = target.to(device)

        node_weights          = aux['node_weights'].to(device)
        directed_edges        = aux['directed_edges'].to(device)
        edge_gradient_weights = aux['edge_gradient_weights'].to(device)
        node_mask             = aux['node_mask'].to(device)

        optimizer.zero_grad()

        pred = model(pos, fx, node_weights, directed_edges,
                     edge_gradient_weights, node_mask)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pos.size(0)
        n_samples  += pos.size(0)

    return total_loss / n_samples


def train_one_step(model, batch, criterion, optimizer, optimizer_inv_L, device):
    model.train()
    pos, fx, target, aux = batch
    pos = pos.to(device)
    fx = fx.to(device)
    target = target.to(device)

    node_weights = aux['node_weights'].to(device)
    directed_edges = aux['directed_edges'].to(device)
    edge_gradient_weights = aux['edge_gradient_weights'].to(device)
    node_mask = aux['node_mask'].to(device)

    optimizer.zero_grad()
    if optimizer_inv_L:
        optimizer_inv_L.zero_grad()

    pred = model(pos, fx, node_weights, directed_edges,
                 edge_gradient_weights, node_mask)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    if optimizer_inv_L:
        optimizer_inv_L.step()
    return float(loss.item())


def validate(model, dataloader, criterion, device, disable_tqdm=False):
    model.eval()
    total_loss   = 0
    n_samples    = 0
    metrics_calc = MetricsCalculator()

    with torch.no_grad():
        for pos, fx, target, aux in tqdm(dataloader, desc="Validation", disable=disable_tqdm):
            pos    = pos.to(device)
            fx     = fx.to(device)
            target = target.to(device)

            node_weights          = aux['node_weights'].to(device)
            directed_edges        = aux['directed_edges'].to(device)
            edge_gradient_weights = aux['edge_gradient_weights'].to(device)
            node_mask             = aux['node_mask'].to(device)

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
    roots = ['/'.join(p[:-2]) if len(p) >= 2 else d for p, d in zip(parts_list, data_dirs)]
    benchmarks = [p[-2] if len(p) >= 2 else '03_damflow' for p in parts_list]
    cases = [p[-1] if len(p) >= 1 else 'case0' for p in parts_list]
    return roots, benchmarks, cases


def _make_single_dataset(dataset_type, data_dir, step_size, train_ratio, seed, split):
    if dataset_type == 'ship':
        ds = IrregularFlowFieldDataset(
            data_dir=data_dir,
            step_size=step_size,
            train_ratio=train_ratio,
            seed=seed,
            normalize=True,
        )
        ds.set_split(split)
        return ds

    if dataset_type == 'cfd_bench':
        roots, benchmarks, cases = _parse_cfd_dirs([data_dir])
        ds = CFDBenchIrregularDataset(
            root=roots[0],
            benchmark=benchmarks[0],
            case=cases[0],
            step_size=step_size,
            train_ratio=train_ratio,
            seed=seed,
            normalize=True,
        )
        ds.set_split(split)
        return ds

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def _make_multi_dataset(dataset_type, data_dirs, step_size, train_ratio, seed, split):
    if dataset_type == 'ship':
        return MultiConditionIrregularDataset(
            data_dirs=data_dirs,
            step_size=step_size,
            train_ratio=train_ratio,
            seed=seed,
            normalize=True,
            split=split,
        )

    if dataset_type == 'cfd_bench':
        roots, benchmarks, cases = _parse_cfd_dirs(data_dirs)
        return MultiConditionCFDBenchIrregularDataset(
            roots=roots,
            benchmarks=benchmarks,
            cases=cases,
            step_size=step_size,
            train_ratio=train_ratio,
            seed=seed,
            normalize=True,
            split=split,
        )

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_datasets(dataset_type, data_dirs, step_size, train_ratio, seed, multi_condition):
    if not multi_condition:
        train_ds = _make_single_dataset(
            dataset_type, data_dirs[0], step_size, train_ratio, seed, 'train'
        )
        val_ds = _make_single_dataset(
            dataset_type, data_dirs[0], step_size, train_ratio, seed, 'test'
        )
        return train_ds, val_ds

    train_ds = _make_multi_dataset(
        dataset_type, data_dirs, step_size, train_ratio, seed, 'train'
    )

    if dataset_type == 'cfd_bench':
        val_ds = MultiConditionCFDBenchIrregularDataset.from_existing(
            train_ds,
            split='test',
            max_points=train_ds.global_max_points,
        )
    else:
        val_ds = _make_multi_dataset(
            dataset_type, data_dirs, step_size, train_ratio, seed, 'test'
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable_tqdm', action='store_true', default=False,
                        help='Disable tqdm progress bars (recommended for log redirection)')
    parser.add_argument('--save_dir',    type=str,
                        default='outputs_pcno', help='Output directory')
    args = parser.parse_args()

    if args.data_dir is not None and args.data_dirs == ['datasets/shipBench/DTC/field/1Re']:
        args.data_dirs = [args.data_dir]
    if len(args.data_dirs) > 1 and not args.multi_condition:
        raise ValueError('Multiple --data_dirs require --multi_condition')

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Multi-condition: {args.multi_condition}")
    print(f"Data dirs: {args.data_dirs}")

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
        args.seed,
        args.multi_condition,
    )

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
            if pos.shape[0] > train_base.global_max_points:
                pos = pos[:train_base.global_max_points]
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
    train_dataset = PCNODataset(train_base, k_neighbors=args.k_neighbors,
                                nmeasures=args.nmeasures)
    print("Building PCNO aux for val split ...")
    val_dataset   = PCNODataset(val_base,   k_neighbors=args.k_neighbors,
                                nmeasures=args.nmeasures)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=pcno_collate_fn, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=pcno_collate_fn, pin_memory=False,
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

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

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
            loss_val = train_one_step(
                model, batch, criterion, optimizer, optimizer_inv_L, device
            )
            scheduler.step()
            train_losses_since_eval.append(loss_val)

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss_val:.6f}")

            if step % args.eval_every != 0 and step != args.max_steps:
                continue

            train_loss = float(np.mean(train_losses_since_eval)) if train_losses_since_eval else loss_val
            train_losses_since_eval = []
            val_loss, val_metrics = validate(model, val_loader, criterion, device, disable_tqdm=disable_tqdm)
            record = {
                'mode': 'step',
                'epoch': -1,
                'step': step,
                'train_loss': train_loss,
                'val_loss': float(val_loss),
                'mae': float(val_metrics['mae']),
                'mse': float(val_metrics['mse']),
                'rmse': float(val_metrics['rmse']),
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

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, disable_tqdm=disable_tqdm)
            if optimizer_inv_L:
                optimizer_inv_L.step()
                optimizer_inv_L.zero_grad()
            print(f"Train Loss: {train_loss:.6f}")

            val_loss, val_metrics = validate(model, val_loader, criterion, device, disable_tqdm=disable_tqdm)
            record = {
                'mode': 'epoch',
                'epoch': epoch + 1,
                'step': -1,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'mae': float(val_metrics['mae']),
                'mse': float(val_metrics['mse']),
                'rmse': float(val_metrics['rmse']),
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
