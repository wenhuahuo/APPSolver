"""
One-shot patch training config test (step-based).

This script is intended for quick training-budget verification before formal runs.
It runs two experiments sequentially:
  1) shipBench (patch + transformer)
  2) cfdBench  (patch + transformer)

Key defaults follow the requested setup:
  - max_steps = 100000
  - eval_every = 200
  - seed = 42

Example:
    python scripts/train_patch_config_test.py \
        --ship_data_dirs datasets/shipBench/DTC/field/1Re \
        --cfd_data_dirs datasets/cfdBench/03_damflow/case0
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.datasets.shipBench import PatchFlowFieldDataset, MultiConditionPatchDataset
from src.datasets.cfdBench import CFDBenchPatchDataset, MultiConditionCFDBenchPatchDataset
from src.models.patch import PatchTransformer, PatchTransformerLoss
from src.core.metrics import patches_to_points, MetricsCalculator


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
    parts_list = [d.split("/") for d in data_dirs]
    roots = ["/".join(p[:-2]) if len(p) >= 2 else d for p, d in zip(parts_list, data_dirs)]
    benchmarks = [p[-2] if len(p) >= 2 else "03_damflow" for p in parts_list]
    cases = [p[-1] if len(p) >= 1 else "case0" for p in parts_list]
    return roots, benchmarks, cases


def _make_single_patch_dataset(
    dataset_type,
    data_dir,
    patch_size,
    output_dim,
    train_ratio,
    seed,
    split,
    use_embedding=False,
):
    if dataset_type == "ship":
        return PatchFlowFieldDataset(
            data_dir=data_dir,
            step_size=1,
            patch_size=patch_size,
            output_dim=output_dim,
            include_coordinates=True,
            normalize=True,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            enable_params=use_embedding,
        )
    if dataset_type == "cfd_bench":
        roots, benchmarks, cases = _parse_cfd_dirs([data_dir])
        return CFDBenchPatchDataset(
            root=roots[0],
            benchmark=benchmarks[0],
            case=cases[0],
            step_size=1,
            patch_size=patch_size,
            output_dim=output_dim,
            include_coordinates=True,
            normalize=True,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
        )
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def _make_multi_patch_dataset(
    dataset_type,
    data_dirs,
    patch_size,
    output_dim,
    train_ratio,
    seed,
    split,
    use_embedding=False,
    global_max_patches=None,
    global_max_points=None,
):
    if dataset_type == "ship":
        return MultiConditionPatchDataset(
            data_dirs=data_dirs,
            step_size=1,
            patch_size=patch_size,
            output_dim=output_dim,
            include_coordinates=True,
            normalize=True,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            enable_params=use_embedding,
            max_patches=global_max_patches,
            max_points=global_max_points,
        )
    if dataset_type == "cfd_bench":
        roots, benchmarks, cases = _parse_cfd_dirs(data_dirs)
        return MultiConditionCFDBenchPatchDataset(
            roots=roots,
            benchmarks=benchmarks,
            cases=cases,
            step_size=1,
            patch_size=patch_size,
            output_dim=output_dim,
            include_coordinates=True,
            normalize=True,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            max_patches=global_max_patches,
            max_points=global_max_points,
        )
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_datasets(
    dataset_type,
    data_dirs,
    patch_size,
    output_dim,
    train_ratio,
    seed,
    multi_condition,
    use_embedding=False,
):
    if not multi_condition:
        train_ds = _make_single_patch_dataset(
            dataset_type,
            data_dirs[0],
            patch_size,
            output_dim,
            train_ratio,
            seed,
            "train",
            use_embedding,
        )
        val_ds = _make_single_patch_dataset(
            dataset_type,
            data_dirs[0],
            patch_size,
            output_dim,
            train_ratio,
            seed,
            "test",
            use_embedding,
        )
        return train_ds, val_ds

    train_ds = _make_multi_patch_dataset(
        dataset_type,
        data_dirs,
        patch_size,
        output_dim,
        train_ratio,
        seed,
        "train",
        use_embedding,
    )

    global_shape = train_ds.get_global_shape()
    g_patches = global_shape["num_patches"]
    g_points = global_shape["max_points"]

    if dataset_type == "cfd_bench":
        val_ds = MultiConditionCFDBenchPatchDataset.from_existing(
            train_ds,
            split="test",
            max_patches=g_patches,
            max_points=g_points,
        )
    else:
        val_ds = _make_multi_patch_dataset(
            dataset_type,
            data_dirs,
            patch_size,
            output_dim,
            train_ratio,
            seed,
            "test",
            use_embedding,
            global_max_patches=g_patches,
            global_max_points=g_points,
        )

    return train_ds, val_ds


def validate(model, dataloader, criterion, device, ref_dataset, use_embedding=False):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    metrics_calc = MetricsCalculator()
    has_quadtree = hasattr(ref_dataset, "quadtree")

    with torch.no_grad():
        for batch in dataloader:
            input_patches = batch["input"].to(device)
            target_patches = batch["output"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)

            params_embed = None
            if use_embedding and "params_embedding" in batch:
                params_embed = batch["params_embedding"].to(device)

            pred = model(input_patches, params_embed=params_embed)
            loss = criterion(pred, target_patches, mask)

            total_loss += loss.item() * input_patches.size(0)
            n_samples += input_patches.size(0)

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
                metrics_calc.update(pred_points, tgt_points, mask=pred_valid & tgt_valid)
            else:
                if mask is not None:
                    bsz, n_patch, n_point = mask.shape
                    c_out = pred.shape[-1] // n_point
                    pred_reshaped = pred.view(bsz, n_patch, n_point, c_out)
                    target_reshaped = target_patches.view(bsz, n_patch, n_point, c_out)
                    valid = mask.unsqueeze(-1).expand_as(pred_reshaped)
                    metrics_calc.update(pred_reshaped[valid], target_reshaped[valid])
                else:
                    metrics_calc.update(pred, target_patches)

    metrics = metrics_calc.compute()
    avg_loss = total_loss / max(1, n_samples)
    return avg_loss, metrics


def _train_one_step(model, batch, criterion, optimizer, device, use_embedding=False):
    model.train()
    input_patches = batch["input"].to(device)
    target_patches = batch["output"].to(device)
    mask = batch.get("mask", None)
    if mask is not None:
        mask = mask.to(device)

    params_embed = None
    if use_embedding and "params_embedding" in batch:
        params_embed = batch["params_embedding"].to(device)

    optimizer.zero_grad()
    pred = model(input_patches, params_embed=params_embed)
    loss = criterion(pred, target_patches, mask)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _init_csv(csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "dataset",
                "step",
                "train_loss_window",
                "val_loss",
                "mae",
                "mse",
                "rmse",
                "elapsed_sec",
            ],
        )
        writer.writeheader()


def _append_csv(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "dataset",
                "step",
                "train_loss_window",
                "val_loss",
                "mae",
                "mse",
                "rmse",
                "elapsed_sec",
            ],
        )
        writer.writerow(row)


def run_experiment(
    dataset_name,
    dataset_type,
    data_dirs,
    multi_condition,
    args,
    device,
    run_dir,
    csv_path,
):
    print("\n" + "=" * 80)
    print(f"Run: {dataset_name}")
    print(f"Data dirs: {data_dirs}")
    print(f"Multi-condition: {multi_condition}")
    print(f"Use embedding: {args.ship_use_embedding if dataset_type == 'ship' else False}")
    print("=" * 80)

    use_embedding = bool(dataset_type == "ship" and args.ship_use_embedding)

    train_dataset, val_dataset = create_datasets(
        dataset_type=dataset_type,
        data_dirs=data_dirs,
        patch_size=args.patch_size,
        output_dim=args.output_dim,
        train_ratio=args.train_ratio,
        seed=args.seed,
        multi_condition=multi_condition,
        use_embedding=use_embedding,
    )

    if multi_condition:
        shape = train_dataset.get_global_shape()
        in_flattened_dim = shape["max_points"] * shape["input_dim"]
        out_flattened_dim = shape["max_points"] * shape["output_dim"]
        print(
            f"Train samples={len(train_dataset)}, val samples={len(val_dataset)}, "
            f"global_shape={shape}"
        )
    else:
        in_flattened_dim = train_dataset.max_points * train_dataset.input_dim
        out_flattened_dim = train_dataset.max_points * train_dataset.output_dim
        print(
            f"Train samples={len(train_dataset)}, val samples={len(val_dataset)}, "
            f"num_patches={train_dataset.num_patches}, max_points={train_dataset.max_points}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PatchTransformer(
        in_flattened_dim=in_flattened_dim,
        out_flattened_dim=out_flattened_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    criterion = PatchTransformerLoss(use_mask=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    step = 0
    epoch = 0
    train_losses_since_eval = []
    records = []
    best = {
        "step": -1,
        "val_loss": float("inf"),
        "mae": None,
        "mse": None,
        "rmse": None,
    }

    train_iter = iter(train_loader)
    t0 = time.time()
    pbar = tqdm(total=args.max_steps, desc=f"{dataset_name} training")

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1
        pbar.update(1)

        train_loss = _train_one_step(
            model,
            batch,
            criterion,
            optimizer,
            device,
            use_embedding=use_embedding,
        )
        train_losses_since_eval.append(train_loss)

        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{train_loss:.6f}", epoch=epoch)

        if step % args.eval_every != 0 and step != args.max_steps:
            continue

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            val_dataset,
            use_embedding=use_embedding,
        )
        train_loss_window = float(np.mean(train_losses_since_eval)) if train_losses_since_eval else train_loss
        elapsed_sec = float(time.time() - t0)
        train_losses_since_eval = []

        record = {
            "run_name": args.run_name,
            "dataset": dataset_name,
            "step": step,
            "train_loss_window": train_loss_window,
            "val_loss": float(val_loss),
            "mae": float(val_metrics.get("mae", float("nan"))),
            "mse": float(val_metrics.get("mse", float("nan"))),
            "rmse": float(val_metrics.get("rmse", float("nan"))),
            "elapsed_sec": elapsed_sec,
        }
        records.append(record)
        _append_csv(csv_path, record)

        print(
            f"[{dataset_name}] step={step:>6d}  "
            f"train_loss={train_loss_window:.6f}  "
            f"val_loss={val_loss:.6f}  "
            f"mae={record['mae']:.6f}  mse={record['mse']:.6f}  rmse={record['rmse']:.6f}"
        )

        if val_loss < best["val_loss"]:
            best.update(
                {
                    "step": step,
                    "val_loss": float(val_loss),
                    "mae": record["mae"],
                    "mse": record["mse"],
                    "rmse": record["rmse"],
                }
            )
            best_path = os.path.join(run_dir, f"{dataset_name}_best.pth")
            torch.save(model.state_dict(), best_path)

    pbar.close()

    final_path = os.path.join(run_dir, f"{dataset_name}_final.pth")
    torch.save(model.state_dict(), final_path)

    summary = {
        "dataset": dataset_name,
        "dataset_type": dataset_type,
        "data_dirs": data_dirs,
        "multi_condition": multi_condition,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "best": best,
        "final_model": final_path,
        "num_records": len(records),
    }
    with open(os.path.join(run_dir, f"{dataset_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, f"{dataset_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="One-shot step-based patch config test")
    parser.add_argument(
        "--ship_data_dirs",
        type=str,
        nargs="+",
        default=["datasets/shipBench/DTC/field/1Re"],
        help="shipBench training paths",
    )
    parser.add_argument(
        "--cfd_data_dirs",
        type=str,
        nargs="+",
        default=["datasets/cfdBench/03_damflow/case0"],
        help="cfdBench training paths",
    )
    parser.add_argument("--ship_multi_condition", action="store_true", default=False)
    parser.add_argument("--cfd_multi_condition", action="store_true", default=False)
    parser.add_argument(
        "--ship_use_embedding",
        action="store_true",
        default=False,
        help="Enable pre-computed ship parameter embedding for shipBench only",
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="outputs_patch_config_test")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = datetime.now().strftime("cfgtest_%Y%m%d_%H%M%S")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Run name: {args.run_name}")

    run_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    csv_path = os.path.join(run_dir, "metrics_all.csv")
    _init_csv(csv_path)

    all_summaries = []
    all_summaries.append(
        run_experiment(
            dataset_name="shipBench",
            dataset_type="ship",
            data_dirs=args.ship_data_dirs,
            multi_condition=args.ship_multi_condition,
            args=args,
            device=device,
            run_dir=run_dir,
            csv_path=csv_path,
        )
    )
    # all_summaries.append(
    #     run_experiment(
    #         dataset_name="cfdBench",
    #         dataset_type="cfd_bench",
    #         data_dirs=args.cfd_data_dirs,
    #         multi_condition=args.cfd_multi_condition,
    #         args=args,
    #         device=device,
    #         run_dir=run_dir,
    #         csv_path=csv_path,
    #     )
    # )

    result_path = os.path.join(run_dir, "summary_all.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nDone.")
    print(f"Config:   {config_path}")
    print(f"Metrics:  {csv_path}")
    print(f"Summary:  {result_path}")


if __name__ == "__main__":
    main()
