"""Create publication figures for the corrected natural-time v2 results."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'axes.linewidth': 0.8,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.3,
    'legend.frameon': False,
    'lines.linewidth': 1.3,
})

HULLS = ['DTC', 'KCS', 'KVLCC2']
METHODS = [
    'app_transformer', 'app_dpt', 'transolver', 'upt',
    'gnot', 'fno', 'fusion_deeponet', 'pcno',
]
DISPLAY = {
    'persistence': 'Persistence',
    'app_transformer': 'APP-Transformer',
    'app_dpt': 'APP-DPT',
    'transolver': 'Transolver',
    'upt': 'UPT',
    'gnot': 'GNOT',
    'fno': 'FNO',
    'fusion_deeponet': 'Fusion-DeepONet',
    'pcno': 'PCNO',
}
COLORS = {
    'persistence': '#272727',
    'app_transformer': '#0F4D92',
    'app_dpt': '#7BA6D1',
    'transolver': '#66668A',
    'upt': '#9A4D8E',
    'gnot': '#B64342',
    'fno': '#A8A8A8',
    'fusion_deeponet': '#C08A6A',
    'pcno': '#42949E',
}
MARKERS = {
    'app_transformer': 'o', 'app_dpt': 's', 'transolver': '^', 'upt': 'v',
    'gnot': 'D', 'fno': 'P', 'fusion_deeponet': 'X', 'pcno': 'h',
}
GFLOPS = {
    'app_transformer': 1.815,
    'app_dpt': 2.047,
    'transolver': 53.644,
    'fno': 7.482,
    'fusion_deeponet': 23.598,
    'upt': 18.936,
    'gnot': 52.705,
    'pcno': 1.694,
}


def panel_label(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='bottom')


def load_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def final_training_mae(run_dir, method):
    metrics = load_json(run_dir / 'metrics.json')
    return metrics['overall']['mae'] if method == 'persistence' else metrics[-1]['mae']


def collect_data(results_root):
    rollout = {}
    one_step = {}
    source_rows = []
    for hull in HULLS:
        hull_root = results_root / hull
        persistence = load_json(
            hull_root / 'persistence' / 'seed42' / 'rollout_metrics.json'
        )['overall']
        one_step[(hull, 'persistence')] = final_training_mae(
            hull_root / 'persistence' / 'seed42', 'persistence'
        )
        for method in METHODS:
            records = load_json(
                hull_root / method / 'seed42' / 'rollout_metrics.json'
            )['overall']
            skills = np.asarray([
                record['mae'] / baseline['mae']
                for record, baseline in zip(records, persistence)
            ], dtype=float)
            rollout[(hull, method)] = skills
            one_step[(hull, method)] = final_training_mae(
                hull_root / method / 'seed42', method
            )
            cumulative = np.sum([record['mae'] for record in records]) / np.sum(
                [record['mae'] for record in persistence]
            )
            for record, baseline, skill in zip(records, persistence, skills):
                source_rows.append({
                    'hull': hull,
                    'method': DISPLAY[method],
                    'horizon': record['horizon'],
                    'mae': record['mae'],
                    'persistence_mae': baseline['mae'],
                    'mae_skill': skill,
                    'cumulative_mae_skill': cumulative,
                })
    return rollout, one_step, source_rows


def save_figure(fig, output_base):
    fig.savefig(output_base.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(output_base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(
        output_base.with_suffix('.tiff'), dpi=600, bbox_inches='tight',
        pil_kwargs={'compression': 'tiff_lzw'},
    )
    fig.savefig(output_base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_rollout(rollout, cumulative, output_dir):
    fig = plt.figure(figsize=(7.2, 4.8))
    grid = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 0.82], hspace=0.48, wspace=0.32
    )
    ymax = 100.0
    legend_handles = []
    legend_labels = []

    for panel_index, hull in enumerate(HULLS):
        ax = fig.add_subplot(grid[0, panel_index])
        horizons = np.arange(1, 51)
        for method in METHODS:
            values = rollout[(hull, method)]
            finite = np.isfinite(values)
            over = np.where(finite & (values > ymax))[0]
            stop = int(over[0]) if len(over) else len(values)
            valid_stop = min(stop + (1 if len(over) else 0), len(values))
            plotted = np.minimum(values[:valid_stop], ymax * 0.94)
            line, = ax.plot(
                horizons[:valid_stop], plotted,
                color=COLORS[method],
                marker=MARKERS[method], markevery=10, markersize=2.7,
                linewidth=1.8 if method in {'app_transformer', 'gnot'} else 1.0,
                alpha=1.0 if method in {'app_transformer', 'gnot'} else 0.82,
                zorder=3 if method in {'app_transformer', 'gnot'} else 2,
            )
            if panel_index == 0:
                legend_handles.append(line)
                legend_labels.append(DISPLAY[method])
            if len(over):
                ax.scatter(horizons[stop], ymax * 0.94, marker='^', s=18,
                           color=COLORS[method], clip_on=False, zorder=5)
        baseline = ax.axhline(1.0, color=COLORS['persistence'], linestyle='--',
                              linewidth=1.2, zorder=1)
        if panel_index == 0:
            legend_handles.insert(0, baseline)
            legend_labels.insert(0, DISPLAY['persistence'])
        ax.set_yscale('log')
        ax.set_xlim(1, 50)
        ax.set_ylim(0.8, ymax)
        ax.set_xticks([1, 10, 25, 50])
        ax.set_title(hull, fontweight='bold')
        ax.set_xlabel('Rollout horizon')
        if panel_index == 0:
            ax.set_ylabel('MAE skill vs persistence')
        ax.tick_params(direction='out', length=2.5, width=0.7)
        panel_label(ax, chr(ord('a') + panel_index))
        if hull == 'KVLCC2':
            ax.text(0.17, 0.96, 'PCNO >100× at h4; NaN at h19',
                    transform=ax.transAxes, ha='left', va='top', fontsize=5.8,
                    color=COLORS['pcno'],
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 1.0})

    ax_heat = fig.add_subplot(grid[1, :])
    masked = np.ma.masked_invalid(cumulative)
    norm = mpl.colors.LogNorm(vmin=1.0, vmax=50.0)
    cmap = mpl.colormaps['Blues'].copy()
    cmap.set_bad('#F2F2F2')
    image = ax_heat.imshow(masked, aspect='auto', cmap=cmap, norm=norm)
    ax_heat.set_xticks(range(len(HULLS)), HULLS)
    ax_heat.set_yticks(range(len(METHODS)), [DISPLAY[m] for m in METHODS])
    ax_heat.tick_params(length=0)
    ax_heat.set_xlabel('Hull group')
    ax_heat.set_title('Cumulative 50-step MAE skill', fontweight='bold', pad=5)
    for i in range(len(METHODS)):
        for j in range(len(HULLS)):
            value = cumulative[i, j]
            text = 'div.' if not np.isfinite(value) else f'{value:.2f}×'
            color = 'white' if np.isfinite(value) and value > 8 else '#272727'
            ax_heat.text(j, i, text, ha='center', va='center',
                         fontsize=6, color=color)
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.025)
    cbar.set_label('Lower is better')
    cbar.set_ticks([1, 2, 5, 10, 20, 50])
    cbar.set_ticklabels(['1', '2', '5', '10', '20', '50'])
    panel_label(ax_heat, 'd')

    fig.legend(legend_handles, legend_labels, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 1.01), columnspacing=1.2, handlelength=2.4)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.12, right=0.94)
    save_figure(fig, output_dir / 'natural_time_v2_rollout')


def geometric_mean_skill(one_step, method):
    ratios = [
        one_step[(hull, method)] / one_step[(hull, 'persistence')]
        for hull in HULLS
    ]
    return math.exp(np.mean(np.log(ratios))), ratios


def plot_efficiency(one_step, timing, output_dir, source_dir):
    timing_by_method = {record['method']: record for record in timing['records']}
    skills = {}
    source_rows = []
    for method in METHODS:
        aggregate, hull_ratios = geometric_mean_skill(one_step, method)
        skills[method] = aggregate
        record = timing_by_method[method]
        row = {
            'method': DISPLAY[method],
            'gflops': GFLOPS[method],
            'gpu_median_ms': record['median_ms'],
            'gpu_q25_ms': record['q25_ms'],
            'gpu_q75_ms': record['q75_ms'],
            'geomean_mae_skill': aggregate,
        }
        row.update({f'{hull}_mae_skill': ratio for hull, ratio in zip(HULLS, hull_ratios)})
        source_rows.append(row)

    with (source_dir / 'efficiency_source.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(source_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x_data = {
        'a': {method: GFLOPS[method] for method in METHODS},
        'b': {method: timing_by_method[method]['median_ms'] for method in METHODS},
    }
    x_labels = {'a': 'Forward computation (GFLOPs)', 'b': 'GPU forward time (ms)'}
    offsets = {
        'app_transformer': (4, 6), 'app_dpt': (4, -10), 'transolver': (-32, -10),
        'upt': (4, 5), 'gnot': (-20, 6), 'fno': (4, 6),
        'fusion_deeponet': (4, -10), 'pcno': (4, 6),
    }

    for panel, ax in zip(['a', 'b'], axes):
        for method in METHODS:
            x = x_data[panel][method]
            y = skills[method]
            marker = 'D' if method.startswith('app_') else 'o'
            ax.scatter(x, y, s=34 if method == 'app_transformer' else 25,
                       marker=marker, color=COLORS[method], edgecolor='white',
                       linewidth=0.6, zorder=3)
            if panel == 'b':
                record = timing_by_method[method]
                ax.errorbar(
                    x, y,
                    xerr=[[x - record['q25_ms']], [record['q75_ms'] - x]],
                    fmt='none', color=COLORS[method], linewidth=0.8,
                    capsize=1.5, zorder=2,
                )
            dx, dy = offsets[method]
            if panel == 'b' and method == 'pcno':
                dx, dy = -25, 5
            if panel == 'b' and method == 'upt':
                dx, dy = 4, 8
            ax.annotate(DISPLAY[method], (x, y), xytext=(dx, dy),
                        textcoords='offset points', fontsize=5.8,
                        fontweight='bold' if method == 'app_transformer' else 'normal',
                        color=COLORS[method])
        ax.axhline(1.0, color=COLORS['persistence'], linestyle='--', linewidth=1.0)
        ax.text(0.5, 1.03, 'Persistence MAE',
                transform=ax.get_yaxis_transform(), ha='center', va='bottom',
                fontsize=5.8, color=COLORS['persistence'],
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 0.5})
        ax.set_xscale('log')
        ax.set_xlabel(x_labels[panel])
        ax.set_ylim(0.88, max(skills.values()) * 1.12)
        ax.tick_params(direction='out', length=2.5, width=0.7)
        panel_label(ax, panel)
    axes[0].set_ylabel('Geometric-mean one-step MAE skill\nacross three hull groups')
    axes[0].set_title('Accuracy–compute trade-off', fontweight='bold')
    axes[1].set_title(
        f"Accuracy–latency trade-off ({timing['gpu']})",
        fontweight='bold',
    )
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.12, right=0.97, wspace=0.25)
    save_figure(fig, output_dir / 'natural_time_v2_efficiency')


def main():
    parser = argparse.ArgumentParser(description='Plot corrected natural-time v2 results')
    parser.add_argument('--results_root', type=Path, required=True)
    parser.add_argument('--timing', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / 'source_data'
    source_dir.mkdir(exist_ok=True)

    rollout, one_step, rollout_rows = collect_data(args.results_root)
    with (source_dir / 'rollout_source.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rollout_rows[0].keys())
        writer.writeheader()
        writer.writerows(rollout_rows)

    cumulative = np.full((len(METHODS), len(HULLS)), np.nan)
    for i, method in enumerate(METHODS):
        for j, hull in enumerate(HULLS):
            rows = [row for row in rollout_rows
                    if row['method'] == DISPLAY[method] and row['hull'] == hull]
            if all(np.isfinite(row['mae']) for row in rows):
                cumulative[i, j] = rows[0]['cumulative_mae_skill']
    plot_rollout(rollout, cumulative, args.output_dir)

    timing = load_json(args.timing)
    plot_efficiency(one_step, timing, args.output_dir, source_dir)


if __name__ == '__main__':
    main()
