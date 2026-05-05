"""
Visualize APP trade-off and knee point from APPTest CSV results.

Generates:
1) app_tradeoff_scatter.png: compression vs error scatter with knee marker
2) app_heatmap_error.png: mean_relative_rmse heatmap over patch_size/downsample_ratio
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_knee(df: pd.DataFrame, practical_error_cap: float = 0.1) -> int:
    """
    Find knee index on non-trivial points (downsample_ratio < 1.0).
    Objective: maximize distance to the line connecting worst-efficiency and worst-error ends
    in normalized (compression, -error) space.
    """
    use = df[
        (df["downsample_ratio"] < 1.0)
        & (df["effective_data_compression_rate"] > 1e-6)
        & (df["mean_relative_rmse"] <= practical_error_cap)
    ].copy()
    if use.empty:
        use = df[(df["downsample_ratio"] < 1.0) & (df["effective_data_compression_rate"] > 1e-6)].copy()
        if use.empty:
            return int(df["score"].idxmax())

    points = use[["padding_compression_ratio", "mean_relative_rmse", "patch_size", "downsample_ratio"]].to_records(index=False)

    front = []
    best_err = np.inf
    for r in sorted(points, key=lambda x: x[0], reverse=True):
        if r[1] < best_err - 1e-12:
            front.append(r)
            best_err = r[1]
    front = front[::-1]

    if len(front) <= 2:
        return int(use.index[0])

    x = np.array([r[0] for r in front], dtype=float)
    y = np.array([r[1] for r in front], dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    pts = np.stack([x_norm, y_norm], axis=1)

    p1 = pts[0]
    p2 = pts[-1]
    line = p2 - p1
    line_norm = np.linalg.norm(line) + 1e-12
    dists = np.abs(np.cross(line, pts - p1)) / line_norm
    front_idx = int(np.argmax(dists))

    picked = front[front_idx]
    mask = (
        (df["patch_size"] == int(picked[2]))
        & (np.isclose(df["downsample_ratio"], float(picked[3])))
    )
    return int(df[mask].index[0])


def plot_scatter(df: pd.DataFrame, out_path: str, knee_idx: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    trivial = df[(df["downsample_ratio"] >= 1.0) | (df["effective_data_compression_rate"] <= 1e-6)]
    non_trivial = df[(df["downsample_ratio"] < 1.0) & (df["effective_data_compression_rate"] > 1e-6)]

    ax.scatter(
        trivial["padding_compression_ratio"],
        trivial["mean_relative_rmse"],
        c="#bbbbbb",
        s=55,
        label="no effective compression",
        alpha=0.8,
        edgecolors="none",
    )

    sc = ax.scatter(
        non_trivial["padding_compression_ratio"],
        non_trivial["mean_relative_rmse"],
        c=non_trivial["patch_size"],
        cmap="viridis",
        s=75,
        label="effective compression",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
    )

    knee = df.loc[knee_idx]
    ax.scatter(
        [knee["padding_compression_ratio"]],
        [knee["mean_relative_rmse"]],
        s=220,
        marker="*",
        c="#ff4d4f",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
        label=f"knee: p={int(knee['patch_size'])}, r={knee['downsample_ratio']}",
    )

    ax.annotate(
        f"knee\npatch={int(knee['patch_size'])}, ratio={knee['downsample_ratio']}",
        xy=(knee["padding_compression_ratio"], knee["mean_relative_rmse"]),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#666", lw=1),
    )

    ax.set_title("APP Trade-off: Compression vs Relative RMSE")
    ax.set_xlabel("Padding Compression Ratio (higher is better)")
    ax.set_ylabel("Mean Relative RMSE (lower is better)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Patch Size")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out_path: str) -> None:
    pivot = df.pivot(index="patch_size", columns="downsample_ratio", values="mean_relative_rmse")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    im = ax.imshow(pivot.values, cmap="magma_r", aspect="auto")

    ax.set_title("Mean Relative RMSE Heatmap")
    ax.set_xlabel("Downsample Ratio")
    ax.set_ylabel("Patch Size")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Relative RMSE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot APP knee point visualizations")
    parser.add_argument("--csv", type=str, required=True, help="APPTest CSV path")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for figures")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["score"] = df["padding_compression_ratio"] / (df["mean_relative_rmse"] + 1e-8)

    out_dir = args.output_dir or os.path.dirname(args.csv)
    os.makedirs(out_dir, exist_ok=True)

    knee_idx = find_knee(df)
    scatter_path = os.path.join(out_dir, "app_tradeoff_scatter.png")
    heatmap_path = os.path.join(out_dir, "app_heatmap_error.png")

    plot_scatter(df, scatter_path, knee_idx)
    plot_heatmap(df, heatmap_path)

    knee = df.loc[knee_idx]
    print("Generated figures:")
    print(f"- {scatter_path}")
    print(f"- {heatmap_path}")
    print("Knee point (non-trivial):")
    print(
        "  patch_size={}, downsample_ratio={}, compression={:.3f}x, mean_relative_rmse={:.6f}".format(
            int(knee["patch_size"]),
            knee["downsample_ratio"],
            knee["padding_compression_ratio"],
            knee["mean_relative_rmse"],
        )
    )


if __name__ == "__main__":
    main()
