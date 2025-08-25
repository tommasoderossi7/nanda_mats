#!/usr/bin/env python3
"""
Generate 10 samples per conflict prompt (stochastic) + deterministic eval toggle.
Saves prompt_id, text, sample_idx, seed, output_text, first-3-token logits, meta.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_global_seed, write_jsonl


def load_model_and_tokenizer(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                           dtype: str = "bf16", device: str = "auto") -> tuple:
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set dtype
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True
    )
    
    return model, tokenizer


def format_prompt(text: str) -> str:
    """Format prompt using Llama-3 chat template."""
    # Simple chat format for Llama-3
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eos_token|><|start_header_id|>assistant<|end_header_id|>\n\n"


def generate_samples(model, tokenizer, prompt_text: str, num_samples: int = 10, 
                    temperature: float = 0.8, top_p: float = 0.95, 
                    max_new_tokens: int = 200, base_seed: int = 42) -> List[Dict]:
    """Generate multiple samples for a given prompt."""
    samples = []
    
    # Format prompt
    formatted_prompt = format_prompt(prompt_text)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
    # Generate samples
    for sample_idx in range(num_samples):
        # Set seed for reproducibility
        sample_seed = base_seed + sample_idx
        set_global_seed(sample_seed)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Extract generated text
        generated_ids = outputs.sequences[0]
        generated_text = tokenizer.decode(generated_ids[input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract first 3 token logits if available
        first_3_logits = None
        if hasattr(outputs, 'scores') and len(outputs.scores) >= 3:
            first_3_logits = []
            for i in range(min(3, len(outputs.scores))):
                logits = outputs.scores[i][0].cpu().numpy()
                first_3_logits.append(logits.tolist())
        
        # Create sample
        sample = {
            "prompt_id": None,  # Will be set by caller
            "text": prompt_text,
            "sample_idx": sample_idx,
            "seed": sample_seed,
            "output_text": generated_text,
            "first_3_token_logits": first_3_logits,
            "meta": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "model_name": model.config._name_or_path,
                "formatted_prompt": formatted_prompt
            }
        }
        samples.append(sample)
    
    return samples


def load_conflict_prompts() -> List[Dict]:
    """Load all conflict prompts from JSON files."""
    prompts_dir = Path("prompts")
    
    # Load F1 and F2 conflicts
    f1_conflicts = json.loads((prompts_dir / "f1_conflicts.json").read_text())
    f2_conflicts = json.loads((prompts_dir / "f2_conflicts.json").read_text())
    
    return f1_conflicts + f2_conflicts


def main():
    parser = argparse.ArgumentParser(description="Generate samples for conflict prompts")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", 
                       help="Model name or path")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"], 
                       help="Model dtype")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--temperature", type=float, default=0.8, 
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, 
                       help="Top-p sampling")
    parser.add_argument("--max_new_tokens", type=int, default=200, 
                       help="Maximum new tokens")
    parser.add_argument("--samples_per_prompt", type=int, default=10, 
                       help="Number of samples per prompt")
    parser.add_argument("--base_seed", type=int, default=42, 
                       help="Base seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", 
                       help="Use deterministic (greedy) generation")
    parser.add_argument("--output", default="data/gens.jsonl", 
                       help="Output file path")
    parser.add_argument("--config_output", default="data/gen_cfg.json", 
                       help="Config output file path")
    parser.add_argument("--repro_prompt_id", help="Prompt ID for REPRO CMD")
    parser.add_argument("--repro_sample_idx", type=int, help="Sample index for REPRO CMD")
    
    args = parser.parse_args()
    
    # Override temperature for deterministic mode
    if args.deterministic:
        args.temperature = 0.0
        print("Using deterministic (greedy) generation")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device)
    
    # Load conflict prompts
    conflict_prompts = load_conflict_prompts()
    print(f"Loaded {len(conflict_prompts)} conflict prompts")
    
    # Generate samples
    all_samples = []
    for prompt in conflict_prompts:
        print(f"Generating samples for {prompt['id']}: {prompt['text'][:50]}...")
        
        samples = generate_samples(
            model, tokenizer, prompt["text"],
            num_samples=args.samples_per_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            base_seed=args.base_seed
        )
        
        # Set prompt_id
        for sample in samples:
            sample["prompt_id"] = prompt["id"]
        
        all_samples.extend(samples)
    
    # Save samples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, all_samples)
    
    # Save configuration
    config = {
        "model": {
            "name": args.model,
            "dtype": args.dtype,
            "device": args.device
        },
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "deterministic": args.deterministic
        },
        "generation": {
            "samples_per_prompt": args.samples_per_prompt,
            "base_seed": args.base_seed,
            "total_prompts": len(conflict_prompts),
            "total_samples": len(all_samples)
        },
        "seed_policy": "base_seed + sample_idx for reproducibility"
    }
    
    config_path = Path(args.config_output)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    
    # Print summary
    print(f"\n📊 Generation Summary:")
    print(f"  Total prompts: {len(conflict_prompts)}")
    print(f"  Samples per prompt: {args.samples_per_prompt}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Output: {output_path}")
    print(f"  Config: {config_path}")
    
    # Print example row
    if all_samples:
        example = all_samples[0]
        print(f"\n📝 Example row:")
        print(f"  prompt_id: {example['prompt_id']}")
        print(f"  sample_idx: {example['sample_idx']}")
        print(f"  text: {example['text'][:50]}...")
        print(f"  output_text: {example['output_text'][:50]}...")
    
    # Print REPRO CMD if requested
    if args.repro_prompt_id and args.repro_sample_idx is not None:
        repro_cmd = f"python src/generate.py --model {args.model} --temperature {args.temperature} --top_p {args.top_p} --max_new_tokens {args.max_new_tokens} --samples_per_prompt {args.samples_per_prompt} --base_seed {args.base_seed} --repro_prompt_id {args.repro_prompt_id} --repro_sample_idx {args.repro_sample_idx}"
        print(f"\n🔄 REPRO CMD for {args.repro_prompt_id} sample {args.repro_sample_idx}:")
        print(f"  {repro_cmd}")


if __name__ == "__main__":
    main() 