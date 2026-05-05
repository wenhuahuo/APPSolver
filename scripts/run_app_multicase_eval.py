"""
Run APP compression/recovery evaluation on multiple cases and plot per-case + average.

For each input file, this script:
1) runs the full patch_size/downsample_ratio grid
2) saves CSV/JSON/Markdown summary
3) saves trade-off scatter + heatmap figures

Then it averages all cases by (patch_size, downsample_ratio), and saves
the same artifacts for the averaged results.
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List

import pandas as pd

from APPTest import (
    evaluate_one_config,
    load_flow_snapshot,
    parse_float_list,
    parse_int_list,
    save_csv,
    save_report,
)
from plot_app_knee import find_knee, plot_heatmap, plot_scatter


def _safe_name(path: str) -> str:
    rel = path.replace("\\", "/")
    parts = [p for p in rel.split("/") if p]
    tail = parts[-4:] if len(parts) >= 4 else parts
    name = "__".join(tail)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_one_case(
    data_file: str,
    output_dir: str,
    patch_sizes: List[int],
    downsample_ratios: List[float],
    output_dim: int,
    downsample_method: str,
    min_points: int,
    enable_distance_refine: bool,
    ship_length: float,
    ref_point_x: float,
    ref_point_y: float,
    distance_threshold_1: float,
    distance_threshold_2: float,
) -> Dict[str, str]:
    _ensure_dir(output_dir)
    coords, flows, channel_names = load_flow_snapshot(data_file, output_dim=output_dim)

    results = []
    total = len(patch_sizes) * len(downsample_ratios)
    done = 0
    print(f"\nCase: {data_file}")
    for patch_size in patch_sizes:
        for ratio in downsample_ratios:
            done += 1
            print(f"  [{done}/{total}] p={patch_size}, r={ratio}")
            row = evaluate_one_config(
                coords=coords,
                flows=flows,
                patch_size=patch_size,
                downsample_ratio=ratio,
                downsample_method=downsample_method,
                min_points=min_points,
                enable_distance_refine=enable_distance_refine,
                ship_length=ship_length,
                ref_point_x=ref_point_x,
                ref_point_y=ref_point_y,
                distance_threshold_1=distance_threshold_1,
                distance_threshold_2=distance_threshold_2,
            )
            results.append(row)

    results_sorted = sorted(results, key=lambda x: x["mean_relative_rmse"])

    csv_path = os.path.join(output_dir, "app_test_results.csv")
    report_path = os.path.join(output_dir, "app_test_report.md")
    summary_path = os.path.join(output_dir, "app_test_summary.json")
    scatter_path = os.path.join(output_dir, "app_tradeoff_scatter.png")
    heatmap_path = os.path.join(output_dir, "app_heatmap_error.png")

    save_csv(results_sorted, csv_path, channel_names)
    save_report(results_sorted, report_path, channel_names)

    df = pd.read_csv(csv_path)
    df["score"] = df["padding_compression_ratio"] / (df["mean_relative_rmse"] + 1e-8)
    knee_idx = find_knee(df)
    plot_scatter(df, scatter_path, knee_idx)
    plot_heatmap(df, heatmap_path)

    knee = df.loc[knee_idx]
    summary = {
        "data_file": data_file,
        "points": int(coords.shape[0]),
        "channels": channel_names,
        "patch_sizes": patch_sizes,
        "downsample_ratios": downsample_ratios,
        "downsample_method": downsample_method,
        "enable_distance_refine": enable_distance_refine,
        "knee": {
            "patch_size": int(knee["patch_size"]),
            "downsample_ratio": float(knee["downsample_ratio"]),
            "padding_compression_ratio": float(knee["padding_compression_ratio"]),
            "mean_relative_rmse": float(knee["mean_relative_rmse"]),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "csv": csv_path,
        "report": report_path,
        "summary": summary_path,
        "scatter": scatter_path,
        "heatmap": heatmap_path,
    }


def _average_cases(case_csvs: List[str], out_dir: str) -> Dict[str, str]:
    _ensure_dir(out_dir)

    dfs = []
    for p in case_csvs:
        d = pd.read_csv(p)
        d["_source"] = os.path.basename(os.path.dirname(p))
        dfs.append(d)

    all_df = pd.concat(dfs, ignore_index=True)
    key_cols = ["patch_size", "downsample_ratio"]

    num_cols = [
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

    avg = all_df.groupby(key_cols, as_index=False)[num_cols].mean()
    avg = avg.sort_values(["mean_relative_rmse", "patch_size", "downsample_ratio"])

    csv_path = os.path.join(out_dir, "app_test_results.csv")
    avg.to_csv(csv_path, index=False)

    report_path = os.path.join(out_dir, "app_test_report.md")
    lines = [
        "# APP Test Report (Average over cases)",
        "",
        "| patch_size | downsample_ratio | compression(padding) | effective_compression_rate | overall_rmse | mean_relative_rmse |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in avg.iterrows():
        lines.append(
            f"| {int(r['patch_size'])} | {float(r['downsample_ratio']):.3f} | {float(r['padding_compression_ratio']):.3f}x | "
            f"{float(r['effective_data_compression_rate']):.3f} | {float(r['overall_rmse']):.6f} | {float(r['mean_relative_rmse']):.6f} |"
        )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    df = pd.read_csv(csv_path)
    df["score"] = df["padding_compression_ratio"] / (df["mean_relative_rmse"] + 1e-8)
    knee_idx = find_knee(df)

    scatter_path = os.path.join(out_dir, "app_tradeoff_scatter.png")
    heatmap_path = os.path.join(out_dir, "app_heatmap_error.png")
    plot_scatter(df, scatter_path, knee_idx)
    plot_heatmap(df, heatmap_path)

    knee = df.loc[knee_idx]
    summary = {
        "num_cases": len(case_csvs),
        "case_csvs": case_csvs,
        "knee": {
            "patch_size": int(knee["patch_size"]),
            "downsample_ratio": float(knee["downsample_ratio"]),
            "padding_compression_ratio": float(knee["padding_compression_ratio"]),
            "mean_relative_rmse": float(knee["mean_relative_rmse"]),
        },
    }
    summary_path = os.path.join(out_dir, "app_test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "csv": csv_path,
        "report": report_path,
        "summary": summary_path,
        "scatter": scatter_path,
        "heatmap": heatmap_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APP test for multiple cases and average")
    parser.add_argument("--data_files", type=str, required=True, help="comma-separated file paths")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--patch_sizes", type=str, default="64,96,128,192,256,384,512,768,1024")
    parser.add_argument(
        "--downsample_ratios",
        type=str,
        default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0",
    )
    parser.add_argument("--output_dim", type=int, default=2)
    parser.add_argument("--downsample_method", type=str, default="distance", choices=["uniform", "distance"])
    parser.add_argument("--min_points", type=int, default=4)
    parser.add_argument("--enable_distance_refine", action="store_true")
    parser.add_argument("--ship_length", type=float, default=7.0)
    parser.add_argument("--ref_point_x", type=float, default=3.0)
    parser.add_argument("--ref_point_y", type=float, default=0.0)
    parser.add_argument("--distance_threshold_1", type=float, default=1.0)
    parser.add_argument("--distance_threshold_2", type=float, default=1.5)
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    patch_sizes = parse_int_list(args.patch_sizes)
    downsample_ratios = parse_float_list(args.downsample_ratios)

    files = [p.strip() for p in args.data_files.split(",") if p.strip()]
    if not files:
        raise ValueError("No data files provided")

    _ensure_dir(args.output_root)

    existing, missing = [], []
    for p in files:
        if os.path.exists(p):
            existing.append(p)
        else:
            missing.append(p)

    if missing and not args.allow_missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    if missing:
        print("Warning: these files are missing and will be skipped:")
        for p in missing:
            print(f"- {p}")

    if not existing:
        raise RuntimeError("No existing data files to run")

    print("=" * 80)
    print("APP multi-case evaluation")
    print("=" * 80)
    print(f"output_root={args.output_root}")
    print(f"num_cases={len(existing)}")

    case_csvs = []
    case_infos = []
    for fpath in existing:
        name = _safe_name(fpath)
        case_dir = os.path.join(args.output_root, name)
        info = _run_one_case(
            data_file=fpath,
            output_dir=case_dir,
            patch_sizes=patch_sizes,
            downsample_ratios=downsample_ratios,
            output_dim=args.output_dim,
            downsample_method=args.downsample_method,
            min_points=args.min_points,
            enable_distance_refine=args.enable_distance_refine,
            ship_length=args.ship_length,
            ref_point_x=args.ref_point_x,
            ref_point_y=args.ref_point_y,
            distance_threshold_1=args.distance_threshold_1,
            distance_threshold_2=args.distance_threshold_2,
        )
        case_csvs.append(info["csv"])
        case_infos.append({"data_file": fpath, **info})

    avg_dir = os.path.join(args.output_root, "average")
    avg_info = _average_cases(case_csvs, avg_dir)

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": args.output_root,
        "num_cases_requested": len(files),
        "num_cases_run": len(existing),
        "missing_files": missing,
        "cases": case_infos,
        "average": avg_info,
    }
    run_summary_path = os.path.join(args.output_root, "run_summary.json")
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Run summary: {run_summary_path}")
    print(f"Average scatter: {avg_info['scatter']}")
    print(f"Average heatmap: {avg_info['heatmap']}")


if __name__ == "__main__":
    main()
