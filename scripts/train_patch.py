"""
Train Patch Models (PatchTransformer / DPT)

Supports single-condition and multi-condition training.
Use --multi_condition to enable multi-condition mode.
Validation set is always derived from the same data_dirs via train/test split.

Usage:
    # Single condition
    python scripts/train_patch.py --model transformer \\
        --data_dirs datasets/shipBench/DTC/field/1Re

    # Multi-condition, validation from test split of training conditions
    python scripts/train_patch.py --model transformer --multi_condition \\
        --data_dirs datasets/shipBench/DTC/field/1Re \\
                   datasets/shipBench/DTC/field/2Re

    # Multi-condition, validation from the same data_dirs split
    python scripts/train_patch.py --model transformer --multi_condition \\
        --data_dirs datasets/shipBench/DTC/field/1Re \\
                   datasets/shipBench/DTC/field/2Re
"""

import os
import sys
import argparse
import random
import csv
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.datasets.shipBench import (
    PatchFlowFieldDataset,
    MultiConditionPatchDataset,
)
from src.datasets.cfdBench import (
    CFDBenchPatchDataset,
    MultiConditionCFDBenchPatchDataset,
)
from src.models.patch import PatchTransformer, PatchTransformerLoss, DPT, DPTLoss
from src.core.metrics import patches_to_points, MetricsCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _parse_cfd_dirs(data_dirs):
    """将 CFDBench 路径列表解析为 (roots, benchmarks, cases)。"""
    parts_list = [d.split('/') for d in data_dirs]
    roots      = ['/'.join(p[:-2]) if len(p) >= 2 else d for p, d in zip(parts_list, data_dirs)]
    benchmarks = [p[-2] if len(p) >= 2 else '03_damflow' for p in parts_list]
    cases      = [p[-1] if len(p) >= 1 else 'case0'      for p in parts_list]
    return roots, benchmarks, cases


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
# Dataset factories
# ---------------------------------------------------------------------------

def _make_single_patch_dataset(dataset_type, data_dir, patch_size, output_dim,
                                 train_ratio, seed, split, use_embedding,
                                 enable_downsample=True, downsample_method='uniform', downsample_ratio=0.25,
                                 embedding_filename='ship_params_embedding.pt', embedding_mode='precomputed',
                                 zero_embedding_dim=0, prefer_cache=True):
    if dataset_type == 'ship':
        return PatchFlowFieldDataset(
            data_dir=data_dir, step_size=1,
            patch_size=patch_size, output_dim=output_dim,
            include_coordinates=True, normalize=True,
            split=split, train_ratio=train_ratio, seed=seed,
            enable_params=use_embedding,
            embedding_filename=embedding_filename,
            embedding_mode=embedding_mode,
            zero_embedding_dim=zero_embedding_dim,
            prefer_cache=prefer_cache,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
        )
    elif dataset_type == 'cfd_bench':
        roots, benchmarks, cases = _parse_cfd_dirs([data_dir])
        return CFDBenchPatchDataset(
            root=roots[0], benchmark=benchmarks[0], case=cases[0],
            step_size=1, patch_size=patch_size, output_dim=output_dim,
            include_coordinates=True, normalize=True,
            split=split, train_ratio=train_ratio, seed=seed,
            enable_params=use_embedding,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
        )
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def _make_multi_patch_dataset(dataset_type, data_dirs, patch_size, output_dim,
                                train_ratio, seed, split, use_embedding,
                                global_max_patches=None, global_max_points=None,
                                enable_downsample=True, downsample_method='uniform', downsample_ratio=0.25,
                                embedding_filename='ship_params_embedding.pt', embedding_mode='precomputed',
                                zero_embedding_dim=0, prefer_cache=True):
    if dataset_type == 'ship':
        return MultiConditionPatchDataset(
            data_dirs=data_dirs, step_size=1,
            patch_size=patch_size, output_dim=output_dim,
            include_coordinates=True, normalize=True,
            split=split, train_ratio=train_ratio, seed=seed,
            enable_params=use_embedding,
            embedding_filename=embedding_filename,
            embedding_mode=embedding_mode,
            zero_embedding_dim=zero_embedding_dim,
            prefer_cache=prefer_cache,
            max_patches=global_max_patches,
            max_points=global_max_points,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
        )
    elif dataset_type == 'cfd_bench':
        roots, benchmarks, cases = _parse_cfd_dirs(data_dirs)
        return MultiConditionCFDBenchPatchDataset(
            roots=roots, benchmarks=benchmarks, cases=cases,
            step_size=1, patch_size=patch_size, output_dim=output_dim,
            include_coordinates=True, normalize=True,
            split=split, train_ratio=train_ratio, seed=seed,
            enable_params=use_embedding,
            max_patches=global_max_patches,
            max_points=global_max_points,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
        )
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_datasets(dataset_type, data_dirs, patch_size, output_dim,
                    train_ratio, seed, use_embedding, multi_condition,
                    enable_downsample=True, downsample_method='uniform', downsample_ratio=0.25,
                    embedding_filename='ship_params_embedding.pt', embedding_mode='precomputed',
                    zero_embedding_dim=0, val_data_dirs=None, val_split='test', prefer_cache=True):
    """
    构建训练集和验证集。

    逻辑:
      - 单工况: data_dirs[0] 按 train_ratio 划分 train/test
      - 多工况: data_dirs 按 train_ratio 分别划分后合并
    """
    if not multi_condition:
        train_ds = _make_single_patch_dataset(
            dataset_type, data_dirs[0], patch_size, output_dim,
            train_ratio, seed, 'train', use_embedding,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
            embedding_filename=embedding_filename,
            embedding_mode=embedding_mode,
            zero_embedding_dim=zero_embedding_dim,
            prefer_cache=prefer_cache)
        val_ds = _make_single_patch_dataset(
            dataset_type, (val_data_dirs or data_dirs)[0], patch_size, output_dim,
            0.0 if val_split == 'all' else train_ratio,
            seed, 'test' if val_split == 'all' else val_split, use_embedding,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
            embedding_filename=embedding_filename,
            embedding_mode=embedding_mode,
            zero_embedding_dim=zero_embedding_dim,
            prefer_cache=prefer_cache)
        return train_ds, val_ds

    # --- 多工况训练集 ---
    train_ds = _make_multi_patch_dataset(
        dataset_type, data_dirs, patch_size, output_dim,
        train_ratio, seed, 'train', use_embedding,
        enable_downsample=enable_downsample,
        downsample_method=downsample_method,
        downsample_ratio=downsample_ratio,
        embedding_filename=embedding_filename,
        embedding_mode=embedding_mode,
        zero_embedding_dim=zero_embedding_dim,
        prefer_cache=prefer_cache)

    # 从训练集获取全局形状，确保验证集 padding 与训练集一致
    global_shape = train_ds.get_global_shape()
    g_patches = global_shape['num_patches']
    g_points  = global_shape['max_points']

    if val_data_dirs is not None:
        val_build_split = 'test' if val_split == 'all' else val_split
        val_train_ratio = 0.0 if val_split == 'all' else train_ratio
        val_probe = _make_multi_patch_dataset(
            dataset_type, val_data_dirs, patch_size, output_dim,
            val_train_ratio, seed, val_build_split, use_embedding,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
            embedding_filename=embedding_filename,
            embedding_mode=embedding_mode,
            zero_embedding_dim=zero_embedding_dim,
            prefer_cache=prefer_cache)
        val_shape = val_probe.get_global_shape()
        g_patches = max(g_patches, val_shape['num_patches'])
        g_points = max(g_points, val_shape['max_points'])
        train_ds = _make_multi_patch_dataset(
            dataset_type, data_dirs, patch_size, output_dim,
            train_ratio, seed, 'train', use_embedding,
            global_max_patches=g_patches,
            global_max_points=g_points,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
            embedding_filename=embedding_filename,
            embedding_mode=embedding_mode,
            zero_embedding_dim=zero_embedding_dim,
            prefer_cache=prefer_cache)
        val_ds = _make_multi_patch_dataset(
            dataset_type, val_data_dirs, patch_size, output_dim,
            val_train_ratio, seed, val_build_split, use_embedding,
            global_max_patches=g_patches,
            global_max_points=g_points,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
            embedding_filename=embedding_filename,
            embedding_mode=embedding_mode,
            zero_embedding_dim=zero_embedding_dim,
            prefer_cache=prefer_cache)
        return train_ds, val_ds

    if dataset_type == 'cfd_bench':
        val_ds = MultiConditionCFDBenchPatchDataset.from_existing(
            train_ds,
            split='test',
            max_patches=g_patches,
            max_points=g_points,
        )
    else:
        val_ds = _make_multi_patch_dataset(
            dataset_type, data_dirs, patch_size, output_dim,
            train_ratio, seed, 'test', use_embedding,
            global_max_patches=g_patches,
            global_max_points=g_points,
            enable_downsample=enable_downsample,
            downsample_method=downsample_method,
            downsample_ratio=downsample_ratio,
            embedding_filename=embedding_filename,
            embedding_mode=embedding_mode,
            zero_embedding_dim=zero_embedding_dim)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device,
                model_type='transformer', use_embedding=False, disable_tqdm=False):
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in tqdm(dataloader, desc="Training", disable=disable_tqdm):
        input_patches  = batch['input'].to(device)
        target_patches = batch['output'].to(device)
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(device)

        params_embed = None
        if use_embedding and 'params_embedding' in batch:
            params_embed = batch['params_embedding'].to(device)

        optimizer.zero_grad()

        if model_type == 'dpt':
            pred = model(input_patches, mask=mask, params_embed=params_embed)
        else:
            pred = model(input_patches, params_embed=params_embed)

        loss = criterion(pred, target_patches, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_patches.size(0)
        n_samples  += input_patches.size(0)

    return total_loss / n_samples


def train_one_step(model, batch, optimizer, criterion, device,
                   model_type='transformer', use_embedding=False):
    model.train()
    input_patches = batch['input'].to(device)
    target_patches = batch['output'].to(device)
    mask = batch.get('mask', None)
    if mask is not None:
        mask = mask.to(device)

    params_embed = None
    if use_embedding and 'params_embedding' in batch:
        params_embed = batch['params_embedding'].to(device)

    optimizer.zero_grad()
    if model_type == 'dpt':
        pred = model(input_patches, mask=mask, params_embed=params_embed)
    else:
        pred = model(input_patches, params_embed=params_embed)

    loss = criterion(pred, target_patches, mask)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def validate(model, dataloader, criterion, device, ref_dataset,
             model_type='transformer', use_embedding=False, disable_tqdm=False):
    """
    验证函数。

    当 ref_dataset 是单工况 PatchFlowFieldDataset 时，额外做
    patches_to_points 转换并计算逐点指标。
    多工况数据集没有统一的 quadtree，只报告 loss。
    """
    model.eval()
    total_loss = 0
    n_samples  = 0
    metrics_calc = MetricsCalculator()

    # 判断是否可以做 patches_to_points（需要单一 quadtree）
    has_quadtree = hasattr(ref_dataset, 'quadtree')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", disable=disable_tqdm):
            input_patches  = batch['input'].to(device)
            target_patches = batch['output'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            params_embed = None
            if use_embedding and 'params_embedding' in batch:
                params_embed = batch['params_embedding'].to(device)

            if model_type == 'dpt':
                pred = model(input_patches, mask=mask, params_embed=params_embed)
            else:
                pred = model(input_patches, params_embed=params_embed)

            loss = criterion(pred, target_patches, mask)
            total_loss += loss.item() * input_patches.size(0)
            n_samples  += input_patches.size(0)

            if has_quadtree:
                pred_points, pred_valid = patches_to_points(
                    patches=pred,
                    quadtree=ref_dataset.quadtree,
                    batch_size=pred.size(0),
                    n_points=ref_dataset.num_points,
                    n_channels=ref_dataset.output_dim,
                    input_dim=ref_dataset.output_dim,
                    max_points=ref_dataset.max_points,
                )
                tgt_points, tgt_valid = patches_to_points(
                    patches=target_patches,
                    quadtree=ref_dataset.quadtree,
                    batch_size=target_patches.size(0),
                    n_points=ref_dataset.num_points,
                    n_channels=ref_dataset.output_dim,
                    input_dim=ref_dataset.output_dim,
                    max_points=ref_dataset.max_points,
                )
                metrics_calc.update(pred_points, tgt_points,
                                    mask=pred_valid & tgt_valid)
            else:
                # 多工况：用 patch 空间直接计算指标（mask 内有效位置）
                if mask is not None:
                    B, P, N = mask.shape
                    C_out = pred.shape[-1] // N
                    pred_reshaped = pred.view(B, P, N, C_out)
                    target_reshaped = target_patches.view(B, P, N, C_out)
                    valid = mask.unsqueeze(-1).expand_as(pred_reshaped)
                    metrics_calc.update(pred_reshaped[valid], target_reshaped[valid])
                else:
                    metrics_calc.update(pred, target_patches)

    metrics = metrics_calc.compute()
    return total_loss / n_samples, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Patch Models')
    parser.add_argument('--model', type=str, default='transformer',
                        choices=['transformer', 'dpt'],
                        help='Model architecture: transformer or dpt')
    parser.add_argument('--dataset_type', type=str, default='ship',
                        choices=['ship', 'cfd_bench'])
    parser.add_argument('--data_dirs', type=str, nargs='+',
                        default=['datasets/shipBench/DTC/field/1Re'],
                        help='Training data path(s). Multiple paths require --multi_condition.')
    parser.add_argument('--val_data_dirs', type=str, nargs='+', default=None,
                        help='Optional validation data path(s), used for leave-one-ship evaluation.')
    parser.add_argument('--val_split', type=str, default='test', choices=['train', 'test', 'all'],
                        help='Validation split when --val_data_dirs is set; all uses every timestep as validation.')
    parser.add_argument('--multi_condition', action='store_true', default=False,
                        help='Enable multi-condition training mode')
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--max_steps',   type=int,   default=0,
                        help='If >0, use step-based training and ignore --epochs')
    parser.add_argument('--eval_every',  type=int,   default=200,
                        help='Validation interval in step-based mode')
    parser.add_argument('--log_every',   type=int,   default=50,
                        help='Logging interval in step-based mode')
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--patch_size',  type=int,   default=64)
    parser.add_argument('--output_dim',  type=int,   default=4,
                        help='Flow output channels')
    parser.add_argument('--d_model',     type=int,   default=64)
    parser.add_argument('--nhead',       type=int,   default=4)
    parser.add_argument('--num_layers',  type=int,   default=4)
    parser.add_argument('--features',    type=int,   default=128,
                        help='DPT feature width')
    parser.add_argument('--n_heads',     type=int,   default=4,
                        help='DPT attention heads')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train/val split ratio')
    parser.add_argument('--enable_downsample', action='store_true', default=True,
                        help='Enable APP downsampling in patch dataset')
    parser.add_argument('--disable_downsample', action='store_true', default=False,
                        help='Disable APP downsampling in patch dataset')
    parser.add_argument('--disable_cache', action='store_true', default=False,
                        help='Load CSV timesteps directly instead of flow_cache.npz')
    parser.add_argument('--downsample_method', type=str, default='distance', choices=['uniform', 'distance'],
                        help='Downsample method used by APP patching')
    parser.add_argument('--downsample_ratio', type=float, default=0.25,
                        help='Patch downsample ratio when downsampling is enabled')
    parser.add_argument('--save_dir',    type=str,   default='outputs_patch')
    parser.add_argument('--use_embedding', action='store_true', default=False,
                        help='Use pre-computed ship parameter embeddings')
    parser.add_argument('--embedding_mode', type=str, default='precomputed', choices=['precomputed', 'zero'],
                        help='Parameter embedding source when --use_embedding is set')
    parser.add_argument('--embedding_filename', type=str, default='ship_params_embedding.pt',
                        help='Pre-computed embedding filename inside each condition directory')
    parser.add_argument('--zero_embedding_dim', type=int, default=0,
                        help='Zero embedding dimension when --embedding_mode zero')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--disable_tqdm', action='store_true', default=False,
                        help='Disable tqdm progress bars (recommended for log redirection)')
    args = parser.parse_args()
    if args.disable_downsample:
        args.enable_downsample = False

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:          {device}")
    print(f"Model:           {args.model}")
    print(f"Dataset type:    {args.dataset_type}")
    print(f"Multi-condition: {args.multi_condition}")
    print(f"Data dirs:       {args.data_dirs}")
    print(f"Val data dirs:   {args.val_data_dirs}")
    print(f"Val split:       {args.val_split if args.val_data_dirs else 'test split from data_dirs'} (ratio={args.train_ratio})")
    print(f"Use embedding:   {args.use_embedding}")
    print(f"Embedding mode:  {args.embedding_mode}, file={args.embedding_filename}, zero_dim={args.zero_embedding_dim}")
    print(f"Downsample:      {args.enable_downsample} ({args.downsample_method}, ratio={args.downsample_ratio})")
    print(f"Prefer cache:    {not args.disable_cache}")

    disable_tqdm = bool(args.disable_tqdm or (not sys.stderr.isatty()))
    print(f"Disable tqdm:    {disable_tqdm}")

    os.makedirs(args.save_dir, exist_ok=True)
    metrics_csv_path = os.path.join(args.save_dir, 'metrics.csv')
    metrics_json_path = os.path.join(args.save_dir, 'metrics.json')
    _init_metrics_csv(metrics_csv_path)
    metrics_records = []
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    print("\nLoading datasets...")
    train_dataset, val_dataset = create_datasets(
        dataset_type=args.dataset_type,
        data_dirs=args.data_dirs,
        patch_size=args.patch_size,
        output_dim=args.output_dim,
        train_ratio=args.train_ratio,
        seed=args.seed,
        use_embedding=args.use_embedding,
        multi_condition=args.multi_condition,
        enable_downsample=args.enable_downsample,
        downsample_method=args.downsample_method,
        downsample_ratio=args.downsample_ratio,
        embedding_filename=args.embedding_filename,
        embedding_mode=args.embedding_mode,
        zero_embedding_dim=args.zero_embedding_dim,
        val_data_dirs=args.val_data_dirs,
        val_split=args.val_split,
        prefer_cache=not args.disable_cache,
    )

    # Determine flattened dims and max_patches for model construction
    if args.multi_condition:
        global_shape = train_dataset.get_global_shape()
        print(f"Train: {len(train_dataset)} samples  "
              f"({train_dataset.num_conditions} conditions), shape={global_shape}")
        print(f"Val:   {len(val_dataset)} samples")
        in_flattened_dim = global_shape['max_points'] * global_shape['input_dim']
        out_flattened_dim = global_shape['max_points'] * global_shape['output_dim']
        global_max_patches = global_shape['num_patches']
    else:
        print(f"Train: {len(train_dataset)} samples  "
              f"patches={train_dataset.num_patches}  max_pts={train_dataset.max_points}")
        print(f"Val:   {len(val_dataset)} samples")
        in_flattened_dim = train_dataset.max_points * train_dataset.input_dim
        out_flattened_dim = train_dataset.max_points * train_dataset.output_dim
        global_max_patches = train_dataset.num_patches * 2

    params_dim = None
    if args.use_embedding:
        args.embedding_dim = int(getattr(train_dataset, 'embedding_dim', 0))
        print(f"Embedding dim: {args.embedding_dim}")
        if args.embedding_dim > 0:
            params_dim = args.embedding_dim
            print(f"Project embedding: {args.embedding_dim} -> d_model {args.d_model}")
        else:
            print("Warning: --use_embedding is set but no pre-computed params_embedding was found.")

    # ------------------------------------------------------------------ #
    # DataLoaders
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    if args.model == 'transformer':
        model = PatchTransformer(
            in_flattened_dim=in_flattened_dim,
            out_flattened_dim=out_flattened_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            params_dim=params_dim,
        ).to(device)
    else:
        model = DPT(
            in_flattened_dim=in_flattened_dim,
            out_flattened_dim=out_flattened_dim,
            features=args.features,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.num_layers,
            max_patches=global_max_patches,
            params_dim=params_dim,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,} total / {n_train:,} trainable")

    if args.model == 'transformer':
        criterion = PatchTransformerLoss(use_mask=True)
    else:
        criterion = DPTLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
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
                model, batch, optimizer, criterion, device,
                model_type=args.model, use_embedding=args.use_embedding,
            )
            train_losses_since_eval.append(loss_val)

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss_val:.6f}")

            if step % args.eval_every != 0 and step != args.max_steps:
                continue

            train_loss = float(np.mean(train_losses_since_eval)) if train_losses_since_eval else loss_val
            train_losses_since_eval = []

            val_loss, val_metrics = validate(
                model, val_loader, criterion, device,
                ref_dataset=val_dataset,
                model_type=args.model, use_embedding=args.use_embedding,
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
                'elapsed_sec': float(time.time() - t0),
            }
            metrics_records.append(record)
            _append_metrics_csv(metrics_csv_path, record)
            print(f"step {step:>6d}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}  "
                  f"MAE: {val_metrics['mae']:.6f}  MSE: {val_metrics['mse']:.6f}  RMSE: {val_metrics['rmse']:.6f}")

            ckpt = os.path.join(args.save_dir, f'model_step_{step}.pth')
            torch.save(model.state_dict(), ckpt)

        pbar.close()
    else:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                model_type=args.model, use_embedding=args.use_embedding,
                disable_tqdm=disable_tqdm,
            )
            print(f"Train Loss: {train_loss:.6f}")

            val_loss, val_metrics = validate(
                model, val_loader, criterion, device,
                ref_dataset=val_dataset,
                model_type=args.model, use_embedding=args.use_embedding,
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
                'elapsed_sec': float(time.time() - t0),
            }
            metrics_records.append(record)
            _append_metrics_csv(metrics_csv_path, record)
            print(f"Val Loss: {val_loss:.6f}  MAE: {val_metrics['mae']:.6f}  "
                  f"MSE: {val_metrics['mse']:.6f}  RMSE: {val_metrics['rmse']:.6f}")

            if (epoch + 1) % 5 == 0:
                ckpt = os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), ckpt)
                print(f"Checkpoint saved: {ckpt}")

    final = os.path.join(args.save_dir, 'model_final.pth')
    torch.save(model.state_dict(), final)
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_records, f, indent=2)
    print(f"\nTraining complete. Model saved to {final}")
    print(f"Validation metrics saved to {metrics_csv_path}")


if __name__ == '__main__':
    main()
