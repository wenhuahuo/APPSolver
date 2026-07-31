"""Evaluate the persistence baseline on corrected ShipBench test pairs."""

import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.metrics import MetricsCalculator
from src.datasets.shipBench import MultiConditionIrregularDataset
from src.datasets.temporal import save_data_protocol


CHANNELS = ['u', 'v', 'w', 'p_rgh']


def evaluate(dataset):
    overall = MetricsCalculator()
    conditions = {}

    for condition_id, sub_dataset in enumerate(dataset.sub_datasets):
        calculator = MetricsCalculator()
        for sample_id in range(len(sub_dataset)):
            _coords, current, target = sub_dataset[sample_id]
            calculator.update(current.unsqueeze(0), target.unsqueeze(0))
            overall.update(current.unsqueeze(0), target.unsqueeze(0))

        condition_name = os.path.basename(os.path.normpath(sub_dataset.data_dir))
        conditions[f'{condition_id}:{condition_name}'] = calculator.compute()

    return {'overall': overall.compute(), 'conditions': conditions}


def main():
    parser = argparse.ArgumentParser(description='ShipBench persistence baseline')
    parser.add_argument('--data_dirs', nargs='+', required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollout_holdout_steps', type=int, default=50)
    parser.add_argument('--save_dir', required=True)
    args = parser.parse_args()

    train_dataset = MultiConditionIrregularDataset(
        data_dirs=args.data_dirs,
        split='train',
        train_ratio=args.train_ratio,
        seed=args.seed,
        rollout_holdout_steps=args.rollout_holdout_steps,
    )
    test_dataset = MultiConditionIrregularDataset(
        data_dirs=args.data_dirs,
        split='test',
        train_ratio=args.train_ratio,
        seed=args.seed,
        rollout_holdout_steps=args.rollout_holdout_steps,
        normalization_params=train_dataset.get_normalization_params(),
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_data_protocol(args.save_dir, train_dataset, test_dataset)
    results = evaluate(test_dataset)
    results['channels'] = CHANNELS

    with open(os.path.join(args.save_dir, 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)

    overall = results['overall']
    with open(os.path.join(args.save_dir, 'metrics.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['mae', 'mse', 'rmse', 'relative_l2'],
        )
        writer.writeheader()
        writer.writerow({key: overall[key] for key in writer.fieldnames})

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
