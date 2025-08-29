#!/usr/bin/env python3
"""
Complete implementation of Step 5a: Build candidate vectors and generate ALL validation responses.
This generates the full 13,250 samples as specified in PLAN.yaml.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
sys.path.append('src')
from utils import set_global_seed, read_jsonl


def extract_post_instruction_positions(formatted_prompt: str, tokenizer: Any) -> List[int]:
    """Extract positions after the instruction (post-assistant start token)."""
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    if assistant_start not in formatted_prompt:
        raise ValueError("Could not find assistant start in prompt")
    
    # Tokenize the full prompt to find assistant start position
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    assistant_start_tokens = tokenizer.encode(assistant_start, add_special_tokens=False)
    
    # Find where assistant section starts
    for i in range(len(prompt_tokens) - len(assistant_start_tokens) + 1):
        if prompt_tokens[i:i+len(assistant_start_tokens)] == assistant_start_tokens:
            # Return positions after assistant header
            return list(range(i + len(assistant_start_tokens), len(prompt_tokens)))
    
    return []


def generate_with_steering(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    steering_vector: np.ndarray,
    layer_idx: int,
    alpha: float,
    mode: str
) -> torch.Tensor:
    """Generate with steering intervention."""
    
    # Convert steering vector to tensor
    steering_tensor = torch.from_numpy(steering_vector).float().to(model.device)
    
    # Get post-instruction positions from the prompt
    formatted_prompt = tokenizer.decode(input_ids[0])
    post_instruction_positions = extract_post_instruction_positions(formatted_prompt, tokenizer)
    
    # Hook function for steering
    def steering_hook(module, input, output):
        hidden_states = output[0]  # [batch, seq_len, d_model]
        batch_size, seq_len, d_model = hidden_states.shape
        
        if mode == "M1":
            # Token-local addition at first 3 assistant tokens with decay [1.0, 0.7, 0.5]
            decays = [1.0, 0.7, 0.5]
            
            # Find assistant start position in current sequence
            if post_instruction_positions:
                assistant_start_pos = min(post_instruction_positions)
                
                for t, decay in enumerate(decays):
                    target_pos = assistant_start_pos + t
                    if target_pos < seq_len:
                        hidden_states[:, target_pos, :] += alpha * decay * steering_tensor
                        
        elif mode == "M2":
            # Layer-wide addition across all post-instruction positions
            if post_instruction_positions:
                for pos in post_instruction_positions:
                    if pos < seq_len:
                        hidden_states[:, pos, :] += alpha * steering_tensor
        
        return (hidden_states,) + output[1:]
    
    # Register hook
    hook = model.model.layers[layer_idx].register_forward_hook(steering_hook)
    
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        return output[0, input_ids.shape[1]:].cpu()
    finally:
        hook.remove()


def generate_responses(
    model: Any,
    tokenizer: Any, 
    prompts: List[Dict],
    gen_lookup: Dict[str, str],
    steering_vector: np.ndarray = None,
    layer_idx: int = None,
    alpha: float = 0.0,
    mode: str = "M1",
    num_samples: int = 10,
    seed_offset: int = 0,
    vector_key: str = None
) -> List[Dict]:
    """Generate responses with optional steering."""
    responses = []
    
    for prompt_idx, prompt_data in enumerate(prompts):
        prompt_id = prompt_data['prompt_id']
        prompt_text = gen_lookup.get(prompt_id, '')
        
        if not prompt_text:
            print(f"Warning: No text found for {prompt_id}")
            continue
            
        # Format prompt using same template as generation
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        for sample_idx in range(num_samples):
            # Set seed for reproducibility
            set_global_seed(42 + seed_offset + prompt_idx * 100 + sample_idx)
            
            # Tokenize input
            input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(model.device)
            
            # Generate with or without steering
            if steering_vector is not None and layer_idx is not None and alpha > 0:
                generated = generate_with_steering(
                    model, tokenizer, input_ids, steering_vector, layer_idx, alpha, mode
                )
            else:
                # Baseline generation (greedy)
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=200,
                        do_sample=False,  # Greedy
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                generated = output[0, input_ids.shape[1]:].cpu()
            
            # Decode response
            response_text = tokenizer.decode(generated, skip_special_tokens=True)
            
            # Store result
            result = {
                'prompt_id': prompt_id,
                'sample_idx': sample_idx,
                'text': prompt_text,
                'response': response_text,
                'steering_params': {
                    'alpha': alpha,
                    'layer_idx': layer_idx,
                    'mode': mode,
                    'vector_key': vector_key
                }
            }
            responses.append(result)
    
    return responses


def main():
    parser = argparse.ArgumentParser(description="Full generation within")
    parser.add_argument("--model_path", default="./models/llama-3.1-8b-instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    
    args = parser.parse_args()
    set_global_seed(args.seed)
    
    print("=== STEP 5A: FULL GENERATION WITHIN ===")
    
    # Load candidate vectors (already created)
    print("Loading candidate vectors...")
    vectors = np.load('artifacts/within/candidate_vectors.npz')
    candidate_vectors = dict(vectors)
    
    # Load metadata for layer indices
    with open('data/acts/within/acts_train_meta.json', 'r') as f:
        meta = json.load(f)
    layer_indices = meta['layer_indices']
    
    print(f"Candidate vectors: {list(candidate_vectors.keys())}")
    
    # Load validation and control sets
    print("Loading validation and control sets...")
    with open('data/splits_within/conflict_validation.json', 'r') as f:
        validation_prompts = json.load(f)
    
    with open('data/splits_within/controls_gold.json', 'r') as f:
        control_prompts = json.load(f)
    
    # Load generation data for prompt text lookup
    print("Loading generation data for prompt texts...")
    gen_data = read_jsonl("data/dev_gens.jsonl")
    gen_lookup = {gen['prompt_id']: gen['text'] for gen in gen_data}
    
    print(f"Validation prompts: {len(validation_prompts)}")
    print(f"Control prompts: {len(control_prompts)}")
    
    # Calculate total work
    modes = ["M1", "M2"]
    alphas = [0.2, 0.4]
    
    baseline_samples = (len(validation_prompts) + len(control_prompts)) * 10
    steered_combinations = len(candidate_vectors) * len(modes) * len(alphas)
    steered_samples = steered_combinations * (len(validation_prompts) + len(control_prompts)) * 10
    total_samples = baseline_samples + steered_samples
    
    print(f"\n=== GENERATION PLAN ===")
    print(f"Baseline samples: {baseline_samples}")
    print(f"Steered combinations: {steered_combinations}")
    print(f"Steered samples: {steered_samples}")
    print(f"Total samples: {total_samples:,}")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    
    print(f"Model loaded on: {model.device}")
    
    # Ensure output directory exists
    Path("data/gens/within").mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # 1. Generate baseline responses
    print("\n=== GENERATING BASELINE RESPONSES ===")
    
    print("Generating baseline validation...")
    baseline_val = generate_responses(
        model, tokenizer, validation_prompts, gen_lookup,
        num_samples=10, seed_offset=0
    )
    
    with open('data/gens/within/baseline_validation_greedy.jsonl', 'w') as f:
        for resp in baseline_val:
            f.write(json.dumps(resp) + '\n')
    
    print("Generating baseline controls...")
    baseline_ctrl = generate_responses(
        model, tokenizer, control_prompts, gen_lookup,
        num_samples=10, seed_offset=1000
    )
    
    with open('data/gens/within/baseline_controls_greedy.jsonl', 'w') as f:
        for resp in baseline_ctrl:
            f.write(json.dumps(resp) + '\n')
    
    print(f"Baseline generation complete: {len(baseline_val) + len(baseline_ctrl)} samples")
    
    # 2. Generate steered responses
    print("\n=== GENERATING STEERED RESPONSES ===")
    
    generation_count = 0
    total_steered_files = steered_combinations * 2
    
    for vector_key, vector in candidate_vectors.items():
        # Extract layer info from key
        if "mid" in vector_key:
            layer_idx = layer_indices['mid']
            layer_name = "mid"
        elif "late" in vector_key:
            layer_idx = layer_indices['late'] 
            layer_name = "late"
        else:
            continue
            
        # Extract timestep
        timestep = vector_key.split('_t')[1] if '_t' in vector_key else "1"
        
        for mode in modes:
            for alpha in alphas:
                generation_count += 1
                print(f"  Progress: {generation_count}/{steered_combinations} - {vector_key}, {mode}, α={alpha}")
                
                # Steered validation
                steered_val = generate_responses(
                    model, tokenizer, validation_prompts, gen_lookup,
                    steering_vector=vector, layer_idx=layer_idx, alpha=alpha, mode=mode,
                    num_samples=10, seed_offset=2000 + generation_count * 1000,
                    vector_key=vector_key
                )
                
                val_filename = f'data/gens/within/steered_val_{layer_name}_t{timestep}_{mode}_a{alpha}.jsonl'
                with open(val_filename, 'w') as f:
                    for resp in steered_val:
                        f.write(json.dumps(resp) + '\n')
                
                # Steered controls
                steered_ctrl = generate_responses(
                    model, tokenizer, control_prompts, gen_lookup,
                    steering_vector=vector, layer_idx=layer_idx, alpha=alpha, mode=mode,
                    num_samples=10, seed_offset=3000 + generation_count * 1000,
                    vector_key=vector_key
                )
                
                ctrl_filename = f'data/gens/within/steered_controls_{layer_name}_t{timestep}_{mode}_a{alpha}.jsonl'
                with open(ctrl_filename, 'w') as f:
                    for resp in steered_ctrl:
                        f.write(json.dumps(resp) + '\n')
                
                # Progress update
                elapsed = time.time() - start_time
                if generation_count % 4 == 0:  # Update every 4 combinations
                    avg_time_per_combo = elapsed / generation_count
                    remaining_combos = steered_combinations - generation_count
                    eta = remaining_combos * avg_time_per_combo
                    print(f"    Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
    
    total_time = time.time() - start_time
    
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total samples generated: {total_samples:,}")
    print(f"Total files created: {2 + total_steered_files}")
    print(f"Average time per sample: {total_time/total_samples:.2f} seconds")
    
    # Verify file count
    import os
    gen_files = [f for f in os.listdir('data/gens/within') if f.endswith('.jsonl')]
    print(f"Files in output directory: {len(gen_files)}")
    
    print(f"\nREPRO CMD: python src/full_generation_within.py --model_path {args.model_path} --seed {args.seed}")


if __name__ == "__main__":
    main()
