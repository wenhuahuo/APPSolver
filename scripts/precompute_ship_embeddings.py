"""
Pre-compute raw LLM embeddings for ship/CFD condition parameters.

This script:
1. Loads ship params from YAML files
2. Encodes them using a frozen LLM (Qwen2.5-0.5B) without projection
3. Saves the embeddings to .pt files for fast loading during training

Usage:
    python scripts/precompute_ship_embeddings.py \
        --data_dir datasets/shipBench/DTC/field/1Re \
        --llm_model Qwen/Qwen2.5-0.5B \
        --output_dir datasets/shipBench/DTC/field/1Re
"""

import os
import sys
import argparse
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.datasets.shipBench import yaml_to_text
from src.models.patch.llm_encoder import LLMEncoder


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-compute LLM embeddings for ship parameters')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Condition directory containing ship_params_*.yaml or cfd_params.yaml')
    parser.add_argument('--llm_model', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='HuggingFace model name for LLM')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for embeddings (default: same as data_dir)')
    parser.add_argument('--params_filename', type=str, default=None,
                        help='Specific params filename (default: auto-detect based on directory)')
    parser.add_argument('--embedding_filename', type=str, default=None,
                        help='Output embedding filename (default inferred from params file)')
    return parser.parse_args()


def find_params_file(data_dir):
    """Auto-detect the condition params YAML file."""
    if os.path.exists(os.path.join(data_dir, 'cfd_params.yaml')):
        return os.path.join(data_dir, 'cfd_params.yaml')
    if os.path.exists(os.path.join(data_dir, 'ship_params_DTC.yaml')):
        return os.path.join(data_dir, 'ship_params_DTC.yaml')
    if os.path.exists(os.path.join(data_dir, 'ship_params_KCS.yaml')):
        return os.path.join(data_dir, 'ship_params_KCS.yaml')
    if os.path.exists(os.path.join(data_dir, 'ship_params_KVLCC2.yaml')):
        return os.path.join(data_dir, 'ship_params_KVLCC2.yaml')
    
    for f in os.listdir(data_dir):
        if f.startswith('ship_params_') and f.endswith('.yaml'):
            return os.path.join(data_dir, f)
    for f in os.listdir(data_dir):
        if f.endswith('.yaml') and 'params' in f:
            return os.path.join(data_dir, f)
    
    raise FileNotFoundError(f"No params YAML found in {data_dir}")


def infer_embedding_filename(params_file, embedding_filename=None):
    if embedding_filename is not None:
        return embedding_filename
    base = os.path.basename(params_file)
    if base == 'cfd_params.yaml':
        return 'cfd_params_embedding.pt'
    return 'ship_params_embedding.pt'


def main():
    args = parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading LLM model: {args.llm_model}")
    llm_encoder = LLMEncoder(
        model_name=args.llm_model,
    )
    llm_encoder.eval()
    
    params_file = args.params_filename or find_params_file(args.data_dir)
    print(f"Loading params from: {params_file}")
    
    with open(params_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    params_text = yaml_to_text(yaml_data)
    print(f"Params text length: {len(params_text)} chars")
    
    print("Encoding with LLM...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding = llm_encoder(params_text, device)
    
    embedding_path = os.path.join(args.output_dir, infer_embedding_filename(params_file, args.embedding_filename))
    torch.save({
        'embedding': embedding.cpu(),
        'params_text': params_text,
        'params_file': params_file,
        'data_dir': args.data_dir,
        'embedding_dim': int(embedding.shape[-1]),
        'llm_hidden_size': int(llm_encoder.hidden_size),
        'llm_model': args.llm_model,
        'projection': 'none',
    }, embedding_path)
    
    print(f"Embedding saved to: {embedding_path}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")


if __name__ == '__main__':
    main()
