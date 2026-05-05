"""
APP compression/recovery test for patch size and downsample ratio.

This script evaluates:
1) data compression amount under different patch_size / downsample_ratio
2) flow-field recovery error after patch downsampling + interpolation restore

Outputs are saved under --output_dir (default: outputs/app_test_<timestamp>):
- app_test_results.csv
- app_test_summary.json
- app_test_report.md
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_processor.mesh_quad import QuadTreeMesh


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_flow_snapshot(path: str, output_dim: int = 2) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "Center:0" not in df.columns or "Center:1" not in df.columns:
            raise ValueError(f"CSV file does not contain Center:0/Center:1 columns: {path}")
        coords = df[["Center:0", "Center:1"]].values.astype(np.float32)

        ship_channels = ["U:0", "U:1", "U:2", "p_rgh"]
        available = [c for c in ship_channels if c in df.columns]
        if not available:
            raise ValueError(f"CSV file does not contain expected flow columns: {path}")
        use_channels = available[:output_dim]
        flows = df[use_channels].values.astype(np.float32)
        return coords, flows, use_channels

    df = pd.read_csv(path, sep=r"\s+")
    alias_map = {
        "volume-fraction-water": "water-vof",
        "y-velocity-water": "y-velocity",
        "x-velocity-water": "x-velocity",
    }
    rename_map = {k: v for k, v in alias_map.items() if k in df.columns and v not in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    if "x-coordinate" not in df.columns or "y-coordinate" not in df.columns:
        raise ValueError(f"TXT file does not contain x-coordinate/y-coordinate columns: {path}")
    coords = df[["x-coordinate", "y-coordinate"]].values.astype(np.float32)

    cfd_channels = ["y-velocity", "x-velocity", "water-vof"]
    available = [c for c in cfd_channels if c in df.columns]
    if not available:
        raise ValueError(f"TXT file does not contain expected flow columns: {path}")
    use_channels = available[:output_dim]
    flows = df[use_channels].values.astype(np.float32)
    return coords, flows, use_channels


def knn_interpolate(sampled_coords: np.ndarray, sampled_values: np.ndarray, query_coords: np.ndarray, k: int = 4) -> np.ndarray:
    if len(sampled_coords) == 0:
        return np.zeros((len(query_coords), sampled_values.shape[1]), dtype=np.float32)

    if len(sampled_coords) == 1:
        return np.repeat(sampled_values, len(query_coords), axis=0)

    k_use = min(k, len(sampled_coords))
    tree = cKDTree(sampled_coords)
    distances, indices = tree.query(query_coords, k=k_use)

    if k_use == 1:
        return sampled_values[indices]

    weights = 1.0 / (distances + 1e-10)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(sampled_values[indices] * weights[..., None], axis=1)


def evaluate_one_config(
    coords: np.ndarray,
    flows: np.ndarray,
    patch_size: int,
    downsample_ratio: float,
    downsample_method: str,
    min_points: int,
    enable_distance_refine: bool,
    ship_length: float,
    ref_point_x: float,
    ref_point_y: float,
    distance_threshold_1: float,
    distance_threshold_2: float,
) -> Dict[str, float]:
    quadtree = QuadTreeMesh(
        coords,
        patch_size=patch_size,
        ship_length=ship_length,
        ref_point=(ref_point_x, ref_point_y),
        distance_threshold_1=distance_threshold_1,
        distance_threshold_2=distance_threshold_2,
        enable_distance_refine=enable_distance_refine,
    )

    original_patch_indices = [patch.points.copy() for patch in quadtree.patches]
    original_counts = np.array([len(idx) for idx in original_patch_indices], dtype=np.float64)
    max_points_before = int(original_counts.max())
    num_patches = len(original_patch_indices)

    target_points = max(min_points, int(patch_size * downsample_ratio))
    quadtree.downsample_patches_by_distance(
        method=downsample_method,
        target_points=target_points,
        min_points=min_points,
    )

    sampled_patch_indices = [patch.points for patch in quadtree.patches]
    sampled_counts = np.array([len(idx) for idx in sampled_patch_indices], dtype=np.float64)
    max_points_after = int(sampled_counts.max())

    storage_before = num_patches * max_points_before
    storage_after = num_patches * max_points_after
    padding_compression_ratio = float(storage_before / storage_after) if storage_after > 0 else 0.0
    mean_point_compression_ratio = float(original_counts.mean() / sampled_counts.mean()) if sampled_counts.mean() > 0 else 0.0
    effective_data_compression_rate = float(1.0 - sampled_counts.mean() / original_counts.mean()) if original_counts.mean() > 0 else 0.0

    recovered = np.zeros_like(flows)
    valid_mask = np.zeros(flows.shape[0], dtype=bool)

    for original_idx, sampled_idx in zip(original_patch_indices, sampled_patch_indices):
        sampled_values = flows[sampled_idx]
        if len(sampled_idx) == len(original_idx):
            recovered_values = sampled_values
        else:
            sampled_coords = coords[sampled_idx]
            original_coords = coords[original_idx]
            recovered_values = knn_interpolate(sampled_coords, sampled_values, original_coords, k=4)

        recovered[original_idx] = recovered_values
        valid_mask[original_idx] = True

    gt = flows[valid_mask]
    pred = recovered[valid_mask]
    rmse_per_channel = np.sqrt(np.mean((gt - pred) ** 2, axis=0))
    overall_rmse = float(np.mean(rmse_per_channel))

    std_per_channel = np.std(gt, axis=0) + 1e-8
    relative_rmse_per_channel = rmse_per_channel / std_per_channel
    mean_relative_rmse = float(np.mean(relative_rmse_per_channel))

    return {
        "patch_size": patch_size,
        "downsample_ratio": downsample_ratio,
        "target_points": target_points,
        "num_patches": num_patches,
        "mean_points_before": float(original_counts.mean()),
        "mean_points_after": float(sampled_counts.mean()),
        "max_points_before": max_points_before,
        "max_points_after": max_points_after,
        "padding_compression_ratio": padding_compression_ratio,
        "mean_point_compression_ratio": mean_point_compression_ratio,
        "effective_data_compression_rate": effective_data_compression_rate,
        "overall_rmse": overall_rmse,
        "mean_relative_rmse": mean_relative_rmse,
        "valid_points": int(valid_mask.sum()),
        "total_points": int(len(valid_mask)),
        "rmse_per_channel": [float(x) for x in rmse_per_channel],
        "relative_rmse_per_channel": [float(x) for x in relative_rmse_per_channel],
    }


def save_csv(results: List[Dict], csv_path: str, channel_names: List[str]) -> None:
    base_fields = [
        "patch_size",
        "downsample_ratio",
        "target_points",
        "num_patches",
        "mean_points_before",
        "mean_points_after",
        "max_points_before",
        "max_points_after",
        "padding_compression_ratio",
        "mean_point_compression_ratio",
        "effective_data_compression_rate",
        "overall_rmse",
        "mean_relative_rmse",
    ]
    fieldnames = base_fields

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {k: item[k] for k in base_fields}
            writer.writerow(row)


def save_report(results: List[Dict], report_path: str, channel_names: List[str]) -> None:
    lines = []
    lines.append("# APP Test Report")
    lines.append("")
    lines.append("| patch_size | downsample_ratio | compression(padding) | effective_compression_rate | overall_rmse | mean_relative_rmse |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['patch_size']} | {r['downsample_ratio']:.3f} | {r['padding_compression_ratio']:.3f}x | "
            f"{r['effective_data_compression_rate']:.3f} | {r['overall_rmse']:.6f} | {r['mean_relative_rmse']:.6f} |"
        )

    best = min(results, key=lambda x: x["mean_relative_rmse"])
    lines.append("")
    lines.append("## Best Config (by mean_relative_rmse)")
    lines.append("")
    lines.append(
        f"- patch_size={best['patch_size']}, downsample_ratio={best['downsample_ratio']}, "
        f"mean_relative_rmse={best['mean_relative_rmse']:.6f}, "
        f"padding_compression_ratio={best['padding_compression_ratio']:.3f}x"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Detailed channel-wise errors are omitted in this simplified report.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="APP patch compression and recovery test")
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/cfdBench/03_damflow/case0/data0-0001.txt",
        help="single snapshot file: CFD txt or ship timestep csv",
    )
    parser.add_argument("--output_dim", type=int, default=2, help="number of flow channels to evaluate")
    parser.add_argument("--patch_sizes", type=str, default="64,128,256,320")
    parser.add_argument("--downsample_ratios", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--downsample_method", type=str, default="distance", choices=["uniform", "distance"])
    parser.add_argument("--min_points", type=int, default=4)
    parser.add_argument("--enable_distance_refine", action="store_true")
    parser.add_argument("--ship_length", type=float, default=7.0)
    parser.add_argument("--ref_point_x", type=float, default=3.0)
    parser.add_argument("--ref_point_y", type=float, default=0.0)
    parser.add_argument("--distance_threshold_1", type=float, default=1.0)
    parser.add_argument("--distance_threshold_2", type=float, default=1.5)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    patch_sizes = parse_int_list(args.patch_sizes)
    downsample_ratios = parse_float_list(args.downsample_ratios)

    if not patch_sizes:
        raise ValueError("patch_sizes cannot be empty")
    if not downsample_ratios:
        raise ValueError("downsample_ratios cannot be empty")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("outputs", f"app_test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    coords, flows, channel_names = load_flow_snapshot(args.data_file, output_dim=args.output_dim)
    if flows.shape[1] < len(channel_names):
        channel_names = channel_names[:flows.shape[1]]

    results: List[Dict] = []

    print("=" * 80)
    print("APP patch compression/recovery test")
    print("=" * 80)
    print(f"data_file={args.data_file}")
    print(f"points={coords.shape[0]}, channels={flows.shape[1]}, channel_names={channel_names}")
    print(f"patch_sizes={patch_sizes}, downsample_ratios={downsample_ratios}")
    print(f"downsample_method={args.downsample_method}, enable_distance_refine={args.enable_distance_refine}")
    print("=" * 80)

    total = len(patch_sizes) * len(downsample_ratios)
    idx = 0
    for patch_size in patch_sizes:
        for ratio in downsample_ratios:
            idx += 1
            print(f"[{idx}/{total}] patch_size={patch_size}, downsample_ratio={ratio}")
            result = evaluate_one_config(
                coords=coords,
                flows=flows,
                patch_size=patch_size,
                downsample_ratio=ratio,
                downsample_method=args.downsample_method,
                min_points=args.min_points,
                enable_distance_refine=args.enable_distance_refine,
                ship_length=args.ship_length,
                ref_point_x=args.ref_point_x,
                ref_point_y=args.ref_point_y,
                distance_threshold_1=args.distance_threshold_1,
                distance_threshold_2=args.distance_threshold_2,
            )
            results.append(result)
            print(
                "  compression={:.3f}x, effective_compression_rate={:.3f}, overall_rmse={:.6f}, mean_relative_rmse={:.6f}".format(
                    result["padding_compression_ratio"],
                    result["effective_data_compression_rate"],
                    result["overall_rmse"],
                    result["mean_relative_rmse"],
                )
            )

    results_sorted = sorted(results, key=lambda x: x["mean_relative_rmse"])
    best = results_sorted[0]

    csv_path = os.path.join(output_dir, "app_test_results.csv")
    summary_path = os.path.join(output_dir, "app_test_summary.json")
    report_path = os.path.join(output_dir, "app_test_report.md")

    save_csv(results_sorted, csv_path, channel_names)
    save_report(results_sorted, report_path, channel_names)

    summary = {
        "data_file": args.data_file,
        "points": int(coords.shape[0]),
        "channels": channel_names,
        "patch_sizes": patch_sizes,
        "downsample_ratios": downsample_ratios,
        "downsample_method": args.downsample_method,
        "enable_distance_refine": args.enable_distance_refine,
        "best_by_mean_relative_rmse": {
            "patch_size": best["patch_size"],
            "downsample_ratio": best["downsample_ratio"],
            "target_points": best["target_points"],
            "num_patches": best["num_patches"],
            "mean_points_before": best["mean_points_before"],
            "mean_points_after": best["mean_points_after"],
            "max_points_before": best["max_points_before"],
            "max_points_after": best["max_points_after"],
            "padding_compression_ratio": best["padding_compression_ratio"],
            "mean_point_compression_ratio": best["mean_point_compression_ratio"],
            "effective_data_compression_rate": best["effective_data_compression_rate"],
            "overall_rmse": best["overall_rmse"],
            "mean_relative_rmse": best["mean_relative_rmse"],
        },
        "all_results_sorted": [
            {
                "patch_size": r["patch_size"],
                "downsample_ratio": r["downsample_ratio"],
                "target_points": r["target_points"],
                "num_patches": r["num_patches"],
                "mean_points_before": r["mean_points_before"],
                "mean_points_after": r["mean_points_after"],
                "max_points_before": r["max_points_before"],
                "max_points_after": r["max_points_after"],
                "padding_compression_ratio": r["padding_compression_ratio"],
                "mean_point_compression_ratio": r["mean_point_compression_ratio"],
                "effective_data_compression_rate": r["effective_data_compression_rate"],
                "overall_rmse": r["overall_rmse"],
                "mean_relative_rmse": r["mean_relative_rmse"],
            }
            for r in results_sorted
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
