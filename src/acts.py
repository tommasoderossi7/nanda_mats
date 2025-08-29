#!/usr/bin/env python3
"""
Extract residual activations for steering vector computation.

This script teacher-forces the first 3 assistant tokens and captures
residual stream activations at mid and late layers for conflict training samples.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import set_global_seed, read_jsonl


def get_layer_indices(model: Any, sites: Dict[str, Any]) -> Dict[str, int]:
    """Map mid/late layer specs to concrete indices based on model depth."""
    num_layers = len(model.model.layers)
    
    # Map symbolic names to concrete indices (~0.4D, ~0.8D)
    indices = {}
    for site in sites['layers']:
        if site == 'mid':
            indices['mid'] = int(0.4 * num_layers)
        elif site == 'late':
            indices['late'] = int(0.8 * num_layers)
        else:
            raise ValueError(f"Unknown layer site: {site}")
    
    print(f"Model has {num_layers} layers")
    print(f"Mapped layer indices: {indices}")
    return indices


def extract_post_instruction_positions(formatted_prompt: str, tokenizer: Any) -> List[int]:
    """Extract positions after the instruction (post-assistant start token)."""
    # Find the assistant start token position
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    if assistant_start not in formatted_prompt:
        raise ValueError("Could not find assistant start in prompt")
    
    # Tokenize the full prompt to find assistant start position
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    assistant_start_tokens = tokenizer.encode(assistant_start, add_special_tokens=False)
    
    # Find where assistant section starts
    for i in range(len(prompt_tokens) - len(assistant_start_tokens) + 1):
        if prompt_tokens[i:i+len(assistant_start_tokens)] == assistant_start_tokens:
            # Position right after assistant header
            return list(range(i + len(assistant_start_tokens), len(prompt_tokens)))
    
    raise ValueError("Could not locate assistant start position in tokenized prompt")


def teacher_force_and_extract(
    model: Any,
    tokenizer: Any,
    formatted_prompt: str,
    target_text: str,
    layer_indices: Dict[str, int],
    timesteps: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Teacher-force the first few tokens and extract activations.
    
    Returns activations dict with keys like 'layer_mid_t1', 'layer_mid_t2', etc.
    """
    # Tokenize input and target
    input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(model.device)
    target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
    
    # Get post-instruction positions for the formatted prompt
    post_instruction_positions = extract_post_instruction_positions(formatted_prompt, tokenizer)
    
    activations = {}
    
    # Teacher-force each timestep
    for t in timesteps:
        if t > len(target_tokens):
            # If target is shorter than requested timesteps, skip
            continue
            
        # Create input with first t target tokens
        teacher_forced_ids = torch.cat([
            input_ids,
            torch.tensor([target_tokens[:t]], device=model.device)
        ], dim=1)
        
        # Forward pass with hooks to capture activations
        layer_activations = {}
        
        def make_hook(layer_name: str):
            def hook(module, input, output):
                # output[0] is the hidden states
                layer_activations[layer_name] = output[0].detach()
            return hook
        
        hooks = []
        for layer_name, layer_idx in layer_indices.items():
            hook = model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_name)
            )
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                _ = model(teacher_forced_ids)
            
            # Extract activations at the target position (last token position)
            target_pos = teacher_forced_ids.shape[1] - 1
            
            for layer_name in layer_indices.keys():
                key = f"layer_{layer_name}_t{t}"
                if layer_name in layer_activations:
                    # Extract activation at target position
                    activations[key] = layer_activations[layer_name][0, target_pos, :].cpu()
                
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
    
    return activations


def load_model_and_tokenizer(model_path: str, device: str = "auto") -> Tuple[Any, Any]:
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"Model loaded on device: {model.device}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Extract activations for steering")
    parser.add_argument("--model_path", default="./models/llama-3.1-8b-instruct",
                       help="Path to model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device for model")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--split_type", default="original", choices=["original", "within"],
                       help="Use original splits or splits_within")
    
    args = parser.parse_args()
    set_global_seed(args.seed)
    
    # Configuration from PLAN.yaml (hard-coded to avoid YAML parsing issues)
    print("Loading configuration...")
    model_config = {
        'name': 'Llama-3.1-8B-Instruct',
        'dtype': 'bf16'
    }
    sites_config = {
        'layers': ['mid', 'late'],
        'timesteps': [1, 2, 3]
    }
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    layer_indices = get_layer_indices(model, sites_config)
    
    # Load training data
    print("Loading training split...")
    if args.split_type == "within":
        train_split_path = "data/splits_within/conflict_train.json"
    else:
        train_split_path = "data/splits/conflict_train.json"
    
    with open(train_split_path, "r") as f:
        train_prompts = json.load(f)
    print(f"Using split: {train_split_path}")
    
    # Load generation data
    print("Loading generation data...")
    gen_data = read_jsonl("data/dev_gens.jsonl")
    gen_lookup = {(item['prompt_id'], item['sample_idx']): item for item in gen_data}
    
    # Load labels
    print("Loading labels...")
    labels = read_jsonl("data/dev_labels_corrected.jsonl")
    label_lookup = {(item['prompt_id'], item['sample_idx']): item for item in labels}
    
    # Collect samples from training prompts
    train_samples = []
    for train_prompt in train_prompts:
        prompt_id = train_prompt['prompt_id']
        for sample_idx in range(10):  # 10 samples per prompt
            key = (prompt_id, sample_idx)
            if key in gen_lookup and key in label_lookup:
                sample = {
                    'prompt_id': prompt_id,
                    'sample_idx': sample_idx,
                    'generation': gen_lookup[key],
                    'label': label_lookup[key]
                }
                train_samples.append(sample)
    
    print(f"Found {len(train_samples)} training samples")
    
    # Extract activations
    print("Extracting activations...")
    all_activations = {}
    metadata = {
        'layer_indices': layer_indices,
        'timesteps': sites_config['timesteps'],
        'model_path': args.model_path,
        'seed': args.seed,
        'samples': []
    }
    
    for i, sample in enumerate(train_samples):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(train_samples)}")
        
        gen_data = sample['generation']
        formatted_prompt = gen_data['meta']['formatted_prompt']
        output_text = gen_data['output_text']
        
        try:
            sample_acts = teacher_force_and_extract(
                model, tokenizer, formatted_prompt, output_text,
                layer_indices, sites_config['timesteps']
            )
            
            # Store activations
            for key, activation in sample_acts.items():
                if key not in all_activations:
                    all_activations[key] = []
                # Convert to float32 to avoid BFloat16 issues with numpy
                all_activations[key].append(activation.float().numpy())
            
            # Store metadata
            metadata['samples'].append({
                'index': i,
                'prompt_id': sample['prompt_id'],
                'sample_idx': sample['sample_idx'],
                'aggregate': sample['label']['aggregate'],
                'type': sample['label']['type']
            })
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    final_activations = {}
    for key, acts_list in all_activations.items():
        if acts_list:  # Only if we have data
            final_activations[key] = np.stack(acts_list)
            print(f"{key}: shape {final_activations[key].shape}")
    
    # Update metadata
    metadata['n_samples'] = len(metadata['samples'])
    if final_activations:
        metadata['d_model'] = next(iter(final_activations.values())).shape[1]
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.split_type == "within":
        acts_path = output_dir / "acts_train_within.npz"
        meta_path = output_dir / "acts_train_meta_within.json"
    else:
        acts_path = output_dir / "acts_train.npz"
        meta_path = output_dir / "acts_train_meta.json"
    
    print(f"Saving activations to {acts_path}")
    np.savez_compressed(acts_path, **final_activations)
    
    print(f"Saving metadata to {meta_path}")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n=== EXTRACTION COMPLETE ===")
    print(f"Extracted activations for {metadata['n_samples']} samples")
    print(f"Model dimension: {metadata.get('d_model', 'N/A')}")
    print(f"Layer indices: {layer_indices}")
    print(f"Activation keys: {list(final_activations.keys())}")
    
    # Verification
    print("\n=== VERIFICATION ===")
    aggregate_counts = {}
    for sample in metadata['samples']:
        agg = sample['aggregate']
        aggregate_counts[agg] = aggregate_counts.get(agg, 0) + 1
    
    print(f"Sample distribution: {aggregate_counts}")
    
    print(f"\nREPRO CMD: python src/acts.py --model_path {args.model_path} --seed {args.seed} --split_type {args.split_type}")


if __name__ == "__main__":
    main()
