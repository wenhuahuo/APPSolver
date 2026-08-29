"""Evaluate autoregressive rollout on the reserved ShipBench rollout window.

Rolls each trained model (and the persistence baseline) forward from the first
frame of the reserved contiguous rollout window and reports per-horizon metrics
against the held-out ground truth frames. Uses the same temporal split, the
same training-set normalization statistics, and the same full-point k-NN
recovery metrics as training/validation (see train_patch.py::validate).

Example:
    python scripts/evaluate_rollout.py \
        --run_root outputs/rebuttal_all_models_corrected_32k/DTC \
        --data_dirs datasets/shipBench/DTC/field/1Re datasets/shipBench/DTC/field/2Re
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.metrics import MetricsCalculator, patches_to_points, recover_points_knn
from src.datasets.shipBench import MultiConditionIrregularDataset, MultiConditionPatchDataset
from src.models.patch.transformer import Transformer
from src.models.patch.dpt import DPT
from src.models.irregular.transolver import Transolver
from src.models.irregular.fno import FNO
from src.models.irregular.fusion_deeponet import FusionDeepONet
from src.models.irregular.upt import UPT
from src.models.irregular.gnot import GNOT
from src.models.irregular.pcno import (
    PCNO, build_aux_from_pos, collate_aux_batch, compute_fourier_modes,
)

CHANNELS = ['u', 'v', 'w', 'p_rgh']
APP_METHODS = ['app_transformer', 'app_dpt']
IRREGULAR_METHODS = ['transolver', 'upt', 'gnot', 'fno', 'fusion_deeponet', 'pcno']
ALL_METHODS = APP_METHODS + IRREGULAR_METHODS + ['persistence']

IRREGULAR_MODELS = {
    'transolver': Transolver,
    'fno': FNO,
    'fusion_deeponet': FusionDeepONet,
    'upt': UPT,
    'gnot': GNOT,
}


# ---------------------------------------------------------------------------
# Model construction from the training run manifest
# ---------------------------------------------------------------------------

def load_run_config(save_dir):
    config_path = os.path.join(save_dir, 'run_config.json')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f'Missing {config_path}; rollout evaluation builds models from the '
            'training run manifest'
        )
    with open(config_path, encoding='utf-8') as handle:
        return json.load(handle)


def build_app_model(method, run_args, global_shape, dpt_max_patches):
    in_dim = global_shape['max_points'] * global_shape['input_dim']
    out_dim = global_shape['max_points'] * global_shape['output_dim']
    params_dim = run_args['embedding_dim'] if run_args.get('use_embedding') else None
    if method == 'app_transformer':
        return Transformer(
            in_flattened_dim=in_dim, out_flattened_dim=out_dim,
            d_model=run_args['d_model'], nhead=run_args['nhead'],
            num_layers=run_args['num_layers'], params_dim=params_dim,
        )
    return DPT(
        in_flattened_dim=in_dim, out_flattened_dim=out_dim,
        features=run_args['features'], d_model=run_args['d_model'],
        n_heads=run_args['n_heads'], n_layers=run_args['num_layers'],
        max_patches=dpt_max_patches, params_dim=params_dim,
        condition_encoder=run_args.get('condition_encoder', 'token'),
    )


def build_irregular_model(method, run_config, n_channels, fourier_modes):
    if method == 'pcno':
        # PCNO is trained by train_irregular_pcno.py, whose manifest stores the
        # CLI args; Fourier modes are geometry-derived and recomputed here.
        run_args = run_config['args']
        return PCNO(
            in_channels=n_channels, out_channels=n_channels,
            modes=fourier_modes, layers=run_args['layers'],
            fc_dim=run_args['fc_dim'], nmeasures=run_args['nmeasures'],
        )
    model_kwargs = run_config['resolved_model_kwargs']
    if model_kwargs.get('fun_dim') not in (None, n_channels):
        raise ValueError(
            f'Condition channel mismatch for {method}: manifest has '
            f"fun_dim={model_kwargs['fun_dim']}, dataset has {n_channels}"
        )
    return IRREGULAR_MODELS[method](**model_kwargs)


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def validate_run_provenance(
    save_dir, config, model_seed, split_seed, data_dirs,
    rollout_holdout_steps, dataset,
):
    config_path = os.path.join(save_dir, 'run_config.json')
    if config.get('model_seed') != model_seed:
        raise ValueError(f'Model seed mismatch between evaluator and {config_path}')
    # train_irregular.py stores 'split_seed' while train_patch.py and
    # train_irregular_pcno.py store 'resolved_split_seed'.
    run_split_seed = config.get('split_seed', config.get('resolved_split_seed'))
    if run_split_seed != split_seed:
        raise ValueError(f'Split seed mismatch between evaluator and {config_path}')
    run_args = config['args']
    if run_args['data_dirs'] != data_dirs:
        raise ValueError(f'Data directory mismatch for {config_path}')
    if run_args['rollout_holdout_steps'] != rollout_holdout_steps:
        raise ValueError(f'Rollout holdout mismatch for {config_path}')

    stats_path = os.path.join(save_dir, 'normalization_stats.npz')
    saved_stats = np.load(stats_path)
    current_stats = dataset.get_normalization_params()
    for key in ('coord_mean', 'coord_std', 'flow_mean', 'flow_std'):
        if not np.array_equal(saved_stats[key], current_stats[key]):
            raise ValueError(f'Normalization mismatch for {key} in {save_dir}')
    return {
        'model_seed': model_seed,
        'split_seed': split_seed,
        'run_config': config_path,
        'normalization_stats': stats_path,
    }


# ---------------------------------------------------------------------------
# Rollout loops
# ---------------------------------------------------------------------------

class HorizonMetrics:
    """One MetricsCalculator per horizon, accumulated over conditions."""

    def __init__(self, horizon, condition_names):
        self.overall = [MetricsCalculator() for _ in range(horizon)]
        self.conditions = {
            name: [MetricsCalculator() for _ in range(horizon)]
            for name in condition_names
        }

    def update(self, step, condition_name, pred, target):
        self.overall[step].update(pred, target)
        self.conditions[condition_name][step].update(pred, target)

    def compute(self):
        def records(calculators):
            out = []
            for step, calc in enumerate(calculators, start=1):
                out.append({'horizon': step, **calc.compute()})
            return out

        return {
            'overall': records(self.overall),
            'conditions': {name: records(calcs) for name, calcs in self.conditions.items()},
        }


def _normalize(values, mean, std):
    return (values - mean) / std


@torch.inference_mode()
def rollout_irregular(model, method, test_dataset, horizon, device, k_neighbors=8):
    flow_mean = torch.from_numpy(test_dataset.normalization_params['flow_mean']).float()
    flow_std = torch.from_numpy(test_dataset.normalization_params['flow_std']).float()
    coord_mean = torch.from_numpy(test_dataset.normalization_params['coord_mean']).float()
    coord_std = torch.from_numpy(test_dataset.normalization_params['coord_std']).float()

    names = [os.path.basename(os.path.normpath(d)) for d in test_dataset.data_dirs]
    names = [f'{i}:{name}' for i, name in enumerate(names)]
    metrics = HorizonMetrics(horizon, names)

    for cond_id, sub_ds in enumerate(test_dataset.sub_datasets):
        coords, flows = sub_ds.get_rollout_sequence()
        if len(coords) < horizon + 1:
            raise ValueError(
                f'Rollout window has {len(coords)} frames, need {horizon + 1}'
            )
        pcno_aux = None
        if method == 'pcno':
            aux = build_aux_from_pos(sub_ds.coords[0], k_neighbors=k_neighbors,
                                     nmeasures=1)
            pcno_aux = {
                key: value.to(device)
                for key, value in collate_aux_batch([aux]).items()
            }

        flow_cur = _normalize(torch.from_numpy(flows[0]).float(), flow_mean, flow_std)
        for step in range(horizon):
            pos = _normalize(torch.from_numpy(coords[step]).float(), coord_mean, coord_std)
            pos = pos.unsqueeze(0).to(device)
            fx = flow_cur.unsqueeze(0).to(device)
            if method == 'pcno':
                pred = model(pos, fx, pcno_aux['node_weights'],
                             pcno_aux['directed_edges'],
                             pcno_aux['edge_gradient_weights'],
                             pcno_aux['node_mask'])
            else:
                pred = model(pos, fx)
            pred = pred[0].cpu()
            target = _normalize(torch.from_numpy(flows[step + 1]).float(),
                                flow_mean, flow_std)
            metrics.update(step, names[cond_id], pred.unsqueeze(0), target.unsqueeze(0))
            flow_cur = pred

    return metrics.compute()


@torch.inference_mode()
def rollout_persistence(test_dataset, horizon):
    flow_mean = torch.from_numpy(test_dataset.normalization_params['flow_mean']).float()
    flow_std = torch.from_numpy(test_dataset.normalization_params['flow_std']).float()

    names = [os.path.basename(os.path.normpath(d)) for d in test_dataset.data_dirs]
    names = [f'{i}:{name}' for i, name in enumerate(names)]
    metrics = HorizonMetrics(horizon, names)

    for cond_id, sub_ds in enumerate(test_dataset.sub_datasets):
        _coords, flows = sub_ds.get_rollout_sequence()
        if len(flows) < horizon + 1:
            raise ValueError(
                f'Rollout window has {len(flows)} frames, need {horizon + 1}'
            )
        pred = _normalize(torch.from_numpy(flows[0]).float(), flow_mean, flow_std)
        pred = pred.unsqueeze(0)
        for step in range(horizon):
            target = _normalize(torch.from_numpy(flows[step + 1]).float(),
                                flow_mean, flow_std)
            metrics.update(step, names[cond_id], pred, target.unsqueeze(0))

    return metrics.compute()


@torch.inference_mode()
def rollout_app(model, method, test_dataset, global_shape, horizon, device):
    """Autoregressive rollout in patch space with full-point k-NN metrics."""
    norm_params = test_dataset.normalization_params
    flow_mean = torch.from_numpy(norm_params['flow_mean']).float()
    flow_std = torch.from_numpy(norm_params['flow_std']).float()
    coord_mean = torch.from_numpy(norm_params['coord_mean']).float()
    coord_std = torch.from_numpy(norm_params['coord_std']).float()

    names = [os.path.basename(os.path.normpath(d)) for d in test_dataset.data_dirs]
    names = [f'{i}:{name}' for i, name in enumerate(names)]
    metrics = HorizonMetrics(horizon, names)

    g_patches = global_shape['num_patches']
    g_points = global_shape['max_points']

    for cond_id, sub_ds in enumerate(test_dataset.sub_datasets):
        input_dim = sub_ds.input_dim
        output_dim = sub_ds.output_dim
        n_patches = sub_ds.num_patches
        max_points = sub_ds.max_points
        patch_points = [patch.points for patch in sub_ds.quadtree.patches]

        coords, flows = sub_ds.get_rollout_sequence()
        if len(coords) < horizon + 1:
            raise ValueError(
                f'Rollout window has {len(coords)} frames, need {horizon + 1}'
            )

        # Initial patch-space flow from the true first frame (padded per patch).
        flow0 = _normalize(torch.from_numpy(flows[0]).float(), flow_mean, flow_std)
        flow_patch = torch.zeros(n_patches, max_points, output_dim)
        for patch_idx, points in enumerate(patch_points):
            flow_patch[patch_idx, :len(points)] = flow0[points]

        mask = torch.zeros(1, g_patches, g_points, dtype=torch.bool, device=device)
        for patch_idx, points in enumerate(patch_points):
            mask[0, patch_idx, :len(points)] = True

        for step in range(horizon):
            coords_step = _normalize(torch.from_numpy(coords[step]).float(),
                                     coord_mean, coord_std)
            input_patches = torch.zeros(1, g_patches, g_points * input_dim)
            for patch_idx, points in enumerate(patch_points):
                n_pts = len(points)
                combined = torch.cat(
                    [coords_step[points], flow_patch[patch_idx, :n_pts]], dim=1
                )
                input_patches[0, patch_idx, :n_pts * input_dim] = combined.flatten()

            if method == 'app_dpt':
                pred = model(input_patches.to(device), mask=mask, params_embed=None)
            else:
                pred = model(input_patches.to(device), params_embed=None)
            pred = pred[0, :n_patches, :max_points * output_dim].cpu()

            sampled_points, _ = patches_to_points(
                pred.unsqueeze(0), sub_ds.quadtree, 1, sub_ds.num_points,
                output_dim, output_dim, max_points,
            )
            recovered = recover_points_knn(
                sampled_points, sub_ds.recovery_indices,
                sub_ds.recovery_weights, sub_ds.sampled_indices,
            )
            target = _normalize(torch.from_numpy(flows[step + 1]).float(),
                                flow_mean, flow_std)
            metrics.update(step, names[cond_id], recovered, target.unsqueeze(0))
            flow_patch = pred.reshape(n_patches, max_points, output_dim)

    return metrics.compute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ShipBench autoregressive rollout evaluation')
    parser.add_argument('--run_root', required=True,
                        help='Run root containing <method>/seed42 checkpoints')
    parser.add_argument('--checkpoint', choices=['final', 'best_mae'], default='final')
    parser.add_argument('--data_dirs', nargs='+', required=True)
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42,
                        help='Model checkpoint seed')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Dataset split/rollout seed (default: --seed)')
    parser.add_argument('--rollout_holdout_steps', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    split_seed = args.seed if args.split_seed is None else args.split_seed
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    methods = args.methods
    need_app = any(m in APP_METHODS for m in methods)
    need_irregular = any(m in IRREGULAR_METHODS + ['persistence'] for m in methods)

    irregular_test = None
    if need_irregular:
        train_ds = MultiConditionIrregularDataset(
            data_dirs=args.data_dirs, split='train', train_ratio=args.train_ratio,
            seed=split_seed, rollout_holdout_steps=args.rollout_holdout_steps,
        )
        irregular_test = MultiConditionIrregularDataset.from_existing(
            train_ds, split='test'
        )

    patch_test = None
    patch_shape = None
    if need_app:
        app_config = load_run_config(
            os.path.join(args.run_root, 'app_transformer', f'seed{args.seed}'))
        app_args = app_config['args']
        train_patch = MultiConditionPatchDataset(
            data_dirs=args.data_dirs, split='train',
            patch_size=app_args['patch_size'],
            enable_downsample=True,
            downsample_method=app_args['downsample_method'],
            downsample_ratio=app_args['downsample_ratio'],
            train_ratio=args.train_ratio,
            seed=split_seed, rollout_holdout_steps=args.rollout_holdout_steps,
        )
        patch_shape = train_patch.get_global_shape()
        patch_test = MultiConditionPatchDataset.from_existing(
            train_patch, split='test'
        )

    checkpoint_name = (
        'model_final.pth' if args.checkpoint == 'final' else 'model_best_mae.pth'
    )
    result_suffix = '' if args.checkpoint == 'final' else '_best_mae'

    for method in methods:
        save_dir = os.path.join(args.run_root, method, f'seed{args.seed}')
        print(f'\n=== {method} ===')
        ckpt = os.path.join(save_dir, checkpoint_name)
        provenance = None
        if method == 'persistence':
            results = rollout_persistence(irregular_test, args.horizon)
        else:
            config = load_run_config(save_dir)
            provenance = validate_run_provenance(
                save_dir, config, args.seed, split_seed, args.data_dirs,
                args.rollout_holdout_steps,
                patch_test if method in APP_METHODS else irregular_test,
            )
            if method in APP_METHODS:
                assert patch_shape is not None
                # train_patch.py uses num_patches * 2 for DPT positional
                # embeddings in single-condition training and the global
                # count otherwise.
                dpt_max_patches = patch_shape['num_patches'] * (
                    2 if len(args.data_dirs) == 1 else 1)
                model = load_checkpoint(
                    build_app_model(method, config['args'], patch_shape,
                                    dpt_max_patches),
                    ckpt, device)
                results = rollout_app(model, method, patch_test, patch_shape,
                                      args.horizon, device)
            elif method == 'pcno':
                assert irregular_test is not None
                run_args = config['args']
                mins = np.full(2, np.inf)
                maxs = np.full(2, -np.inf)
                for sub_ds in irregular_test.sub_datasets:
                    pos = sub_ds.coords[0]
                    mins = np.minimum(mins, pos.min(axis=0))
                    maxs = np.maximum(maxs, pos.max(axis=0))
                fourier_modes = compute_fourier_modes(
                    2, [run_args['n_modes']] * 2,
                    ((maxs - mins) + 1e-6).tolist(),
                )
                n_channels = irregular_test.sub_datasets[0].n_channels
                model = load_checkpoint(
                    build_irregular_model(
                        method, config, n_channels, fourier_modes),
                    ckpt, device)
                results = rollout_irregular(
                    model, method, irregular_test, args.horizon, device,
                    k_neighbors=run_args['k_neighbors'],
                )
            else:
                assert irregular_test is not None
                n_channels = irregular_test.sub_datasets[0].n_channels
                model = load_checkpoint(
                    build_irregular_model(
                        method, config, n_channels, fourier_modes=None),
                    ckpt, device,
                )
                results = rollout_irregular(model, method, irregular_test,
                                            args.horizon, device)

        results['channels'] = CHANNELS
        results['checkpoint'] = args.checkpoint
        if provenance is not None:
            results['provenance'] = provenance
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(
            save_dir, f'rollout_metrics{result_suffix}.json'
        )
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        with open(json_path, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)

        csv_path = os.path.join(
            save_dir, f'rollout_metrics{result_suffix}.csv'
        )
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=['horizon', 'mae', 'mse', 'rmse', 'relative_l2'])
            writer.writeheader()
            for record in results['overall']:
                writer.writerow({key: record[key] for key in writer.fieldnames})

        final = results['overall'][-1]
        print(f'  saved {json_path}')
        print(f'  horizon {final["horizon"]}: mae={final["mae"]:.4f} '
              f'rmse={final["rmse"]:.4f} rel_l2={final["relative_l2"]:.4f}')


if __name__ == '__main__':
    main()
