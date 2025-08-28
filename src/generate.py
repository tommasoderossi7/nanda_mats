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


def load_model_and_tokenizer(model_name: str = "./models/llama-3.1-8b-instruct", 
                           dtype: str = "bf16", device: str = "auto") -> tuple:
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set dtype
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        local_files_only=True
    )
    
    return model, tokenizer


def format_prompt(tokenizer, text: str) -> str:
    """Format prompt using Llama-3.1 chat template."""
    messages = [
        {"role": "user", "content": text}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_samples(model, tokenizer, prompt_text: str, num_samples: int = 10, 
                    temperature: float = 0.8, top_p: float = 0.95, 
                    max_new_tokens: int = 200, base_seed: int = 42, 
                    save_logits: bool = True) -> tuple:
    """Generate multiple samples for a given prompt.
    
    Returns:
        tuple: (samples, logits_data) where logits_data is list of logit entries
    """
    samples = []
    logits_data = []
    
    # Format prompt
    formatted_prompt = format_prompt(tokenizer, prompt_text)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generate samples
    for sample_idx in range(num_samples):
        # Set seed for reproducibility
        sample_seed = base_seed + sample_idx
        set_global_seed(sample_seed)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=save_logits
            )
        
        # Extract generated text
        generated_ids = outputs.sequences[0]
        generated_text = tokenizer.decode(generated_ids[input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract first 3 token logits if available and requested
        if save_logits and hasattr(outputs, 'scores') and len(outputs.scores) >= 3:
            first_3_logits = []
            for i in range(min(3, len(outputs.scores))):
                logits = outputs.scores[i][0].cpu().numpy()
                first_3_logits.append(logits.tolist())
            
            # Create logits entry (will be linked by prompt_id + sample_idx)
            logits_entry = {
                "prompt_id": None,  # Will be set by caller
                "sample_idx": sample_idx,
                "first_3_token_logits": first_3_logits
            }
            logits_data.append(logits_entry)
        
        # Create sample (without logits)
        sample = {
            "prompt_id": None,  # Will be set by caller
            "text": prompt_text,
            "sample_idx": sample_idx,
            "seed": sample_seed,
            "output_text": generated_text,
            "meta": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "model_name": model.config._name_or_path,
                "formatted_prompt": formatted_prompt
            }
        }
        samples.append(sample)
    
    return samples, logits_data


def load_conflict_prompts(conflict_families: Optional[List[str]] = None, 
                          specific_prompts: Optional[List[str]] = None) -> List[Dict]:
    """Load conflict prompts from JSON files with optional filtering."""
    prompts_dir = Path("prompts")
    all_prompts = []
    
    # Load all prompt files
    prompt_files = {
        "f1": ["f1_conflicts.json", "f1_nonconf_minpairs.json"],
        "f2": ["f2_conflicts.json", "f2_nonconf_minpairs.json"],
        "benign": ["benign.json"]
    }
    
    # Determine which families to load
    families_to_load = conflict_families if conflict_families else ["f1", "f2", "benign"]
    
    # Load prompts from specified families
    for family in families_to_load:
        if family in prompt_files:
            for filename in prompt_files[family]:
                file_path = prompts_dir / filename
                if file_path.exists():
                    prompts = json.loads(file_path.read_text())
                    all_prompts.extend(prompts)
    
    # Filter by specific prompt IDs if provided
    if specific_prompts:
        all_prompts = [p for p in all_prompts if p["id"] in specific_prompts]
    
    return all_prompts


def main():
    parser = argparse.ArgumentParser(description="Generate samples for conflict prompts")
    parser.add_argument("--model", default="./models/llama-3.1-8b-instruct", 
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
    parser.add_argument("--logits_output", default="data/logits.jsonl", 
                       help="Logits output file path")
    parser.add_argument("--config_output", default="data/gen_cfg.json", 
                       help="Config output file path")
    parser.add_argument("--save_logits", action="store_true", default=True,
                       help="Save first 3 token logits (default: True)")
    parser.add_argument("--no_logits", action="store_true", 
                       help="Don't save logits (overrides --save_logits)")
    parser.add_argument("--repro_prompt_id", help="Prompt ID for REPRO CMD")
    parser.add_argument("--repro_sample_idx", type=int, help="Sample index for REPRO CMD")
    parser.add_argument("--conflict_family", 
                       help="Comma-separated list of conflict families to run (e.g., f1,f2)")
    parser.add_argument("--prompts_to_run", 
                       help="Comma-separated list of specific prompt IDs to run (e.g., f1_002_nonconf,f2_001)")
    parser.add_argument("--prompt_string", type=str, 
                       help="A custom prompt string to generate samples for")
    
    args = parser.parse_args()
    
    # Override temperature for deterministic mode
    if args.deterministic:
        args.temperature = 0.0
        print("Using deterministic (greedy) generation")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device)
    
    # Parse filtering arguments
    conflict_families = None
    if args.conflict_family:
        conflict_families = [f.strip() for f in args.conflict_family.split(",")]
        print(f"Filtering by conflict families: {conflict_families}")
    
    specific_prompts = None
    if args.prompts_to_run:
        specific_prompts = [p.strip() for p in args.prompts_to_run.split(",")]
        print(f"Filtering by specific prompts: {specific_prompts}")
    
    # Load conflict prompts
    conflict_prompts = load_conflict_prompts(conflict_families, specific_prompts)
    print(f"Loaded {len(conflict_prompts)} conflict prompts")
    
    # Determine if saving logits
    save_logits = args.save_logits and not args.no_logits
    if args.no_logits:
        print("Logits saving disabled")
    elif save_logits:
        print(f"Logits will be saved to: {args.logits_output}")
    
    # Generate samples
    all_samples = []
    all_logits = []
    
    if args.prompt_string:
        print(f"Generating samples for custom prompt: {args.prompt_string[:50]}...")
        prompt_id = "custom_prompt"
        samples, logits_data = generate_samples(
            model, tokenizer, args.prompt_string,
            num_samples=args.samples_per_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            base_seed=args.base_seed,
            save_logits=save_logits
        )
        for sample in samples:
            sample["prompt_id"] = prompt_id
        for logits_entry in logits_data:
            logits_entry["prompt_id"] = prompt_id
        all_samples.extend(samples)
        all_logits.extend(logits_data)
        output_filename = "exp_gens.jsonl"
        logits_filename = "exp_logits.jsonl"
        config_filename = "exp_gen_cfg.json"
    else:
        for prompt in conflict_prompts:
            print(f"Generating samples for {prompt['id']}: {prompt['text'][:50]}...")
            
            samples, logits_data = generate_samples(
                model, tokenizer, prompt["text"],
                num_samples=args.samples_per_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                base_seed=args.base_seed,
                save_logits=save_logits
            )
            
            # Set prompt_id for both samples and logits
            for sample in samples:
                sample["prompt_id"] = prompt["id"]
            for logits_entry in logits_data:
                logits_entry["prompt_id"] = prompt["id"]
            
            all_samples.extend(samples)
            all_logits.extend(logits_data)
        output_filename = "gens.jsonl"
        logits_filename = "logits.jsonl"
        config_filename = "gen_cfg.json"
    
    # Save samples
    output_path = Path(f"data/{output_filename}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, all_samples)
    
    # Save logits if any were collected
    if all_logits:
        logits_path = Path(f"data/{logits_filename}")
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(logits_path, all_logits)
        print(f"Saved {len(all_logits)} logits entries to: {logits_path}")
    
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
            "total_prompts": len(conflict_prompts) if not args.prompt_string else 1,
            "total_samples": len(all_samples),
            "save_logits": save_logits,
            "total_logits": len(all_logits)
        },
        "seed_policy": "base_seed + sample_idx for reproducibility"
    }
    
    config_path = Path(f"data/{config_filename}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    
    # Print summary
    print(f"\n📊 Generation Summary:")
    print(f"  Total prompts: {len(conflict_prompts) if not args.prompt_string else 1}")
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