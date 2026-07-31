"""Pre-compute frozen-LLM condition embeddings for ShipBench."""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.datasets.shipBench import find_ship_params_file, yaml_to_text
from src.models.patch.llm_encoder import LLMEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pre-compute LLM embeddings for ship parameters'
    )
    parser.add_argument('--data_dirs', nargs='+', required=True)
    parser.add_argument('--llm_model', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--embedding_filename', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {args.llm_model} on {device}')
    encoder = LLMEncoder(args.llm_model).eval()

    for data_dir in args.data_dirs:
        params_file = find_ship_params_file(data_dir)
        with open(params_file, 'r', encoding='utf-8') as handle:
            params_text = yaml_to_text(yaml.safe_load(handle))
        embedding = encoder(params_text, device).cpu()
        output_path = os.path.join(data_dir, args.embedding_filename)
        torch.save({
            'embedding': embedding,
            'params_text': params_text,
            'params_file': params_file,
            'data_dir': data_dir,
            'embedding_dim': int(embedding.shape[-1]),
            'llm_hidden_size': int(encoder.hidden_size),
            'llm_model': args.llm_model,
            'pooling': 'last_non_padding_token',
            'projection': 'none',
        }, output_path)
        print(f'Saved {output_path}: shape={tuple(embedding.shape)}')


if __name__ == '__main__':
    main()
