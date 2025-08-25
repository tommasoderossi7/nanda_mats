#!/usr/bin/env python3
"""
Mock generation script for testing - simulates 10 samples per conflict prompt.
This version doesn't require the actual model and creates realistic mock outputs.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_global_seed, write_jsonl


def load_conflict_prompts() -> List[Dict]:
    """Load all conflict prompts from JSON files."""
    prompts_dir = Path("prompts")
    
    # Load F1 and F2 conflicts
    f1_conflicts = json.loads((prompts_dir / "f1_conflicts.json").read_text())
    f2_conflicts = json.loads((prompts_dir / "f2_conflicts.json").read_text())
    
    return f1_conflicts + f2_conflicts


def mock_generate_samples(
    prompt_text: str,
    prompt_id: str,
    num_samples: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 200,
    base_seed: int = 42
) -> List[Dict]:
    """Generate mock samples that simulate realistic responses to conflict prompts."""
    samples = []
    
    # Set seed for reproducibility
    random.seed(base_seed)
    
    # Define response patterns based on prompt family
    if prompt_id.startswith("f1_"):
        # Formatting conflicts - Type-2 responses (refusal + clarification)
        type2_responses = [
            "I cannot create both formats simultaneously as they have incompatible requirements. Which format would you prefer - CSV or JSON?",
            "There's a conflict between the two formatting constraints. Should I prioritize one format over the other?",
            "I cannot satisfy both formatting requirements at once. Which format is more important for your needs?",
            "This creates an impossible situation with the two formats. Could you clarify which format you'd like me to use?",
            "I cannot format the data as both CSV and JSON simultaneously. Which format would work better for your use case?",
            "There's a contradiction between the formatting requirements. Which format should I focus on?",
            "I cannot meet both formatting constraints. What's your preferred format for this data?",
            "This requires choosing between incompatible formats. Which one is your priority?",
            "I cannot format the same data in both ways at once. Which format do you need?",
            "There's a conflict between the two formatting needs. Which format is more suitable?"
        ]
        # Type-1 responses (refusal only)
        type1_responses = [
            "I cannot create both formats simultaneously.",
            "This is impossible to do with both formats.",
            "I cannot satisfy both formatting requirements.",
            "This creates an impossible formatting situation.",
            "I cannot format the data in both ways at once.",
            "This is not possible with the given constraints.",
            "I cannot meet both formatting requirements.",
            "This creates a formatting conflict.",
            "I cannot format the same data in both formats.",
            "This is impossible with the current constraints."
        ]
        # Type-3 responses (pick one, no acknowledgement)
        type3_responses = [
            "I'll create a CSV file with the data.",
            "Here's the data in JSON format.",
            "I'll format this as a table.",
            "Here's the information in paragraph format.",
            "I'll present this as a flowchart.",
            "Here's the data in timeline format.",
            "I'll create a poem about this topic.",
            "Here's a technical report on the subject.",
            "I'll format this as a pie chart.",
            "Here's the data as a bar graph."
        ]
    else:
        # Word-count conflicts - Type-2 responses
        type2_responses = [
            "I cannot provide both a brief summary and detailed explanation simultaneously. Which length would you prefer?",
            "There's a conflict between the word count requirements. Should I focus on brevity or detail?",
            "I cannot satisfy both length constraints at once. Which approach would work better for you?",
            "This creates an impossible situation with the word limits. Could you clarify your preference?",
            "I cannot write both a short and long version simultaneously. Which length is more important?",
            "There's a contradiction between the length requirements. Which should I prioritize?",
            "I cannot meet both word count constraints. What's your preferred level of detail?",
            "This requires choosing between brief and comprehensive. Which do you need?",
            "I cannot provide both lengths at once. Which would be more useful?",
            "There's a conflict between the length needs. Which approach suits you better?"
        ]
        # Type-1 responses
        type1_responses = [
            "I cannot provide both lengths simultaneously.",
            "This is impossible with both word count requirements.",
            "I cannot satisfy both length constraints.",
            "This creates an impossible length situation.",
            "I cannot write both versions at once.",
            "This is not possible with the given constraints.",
            "I cannot meet both length requirements.",
            "This creates a length conflict.",
            "I cannot provide both versions simultaneously.",
            "This is impossible with the current constraints."
        ]
        # Type-3 responses
        type3_responses = [
            "Here's a brief summary of the topic.",
            "I'll provide a detailed explanation.",
            "Here's a concise overview.",
            "I'll give you a comprehensive analysis.",
            "Here's a short description.",
            "I'll provide an extensive explanation.",
            "Here's a brief answer to your question.",
            "I'll give you a thorough response.",
            "Here's a quick summary.",
            "I'll provide a detailed answer."
        ]
    
    # Combine all response types
    all_responses = type2_responses + type1_responses + type3_responses
    
    # Generate samples
    for sample_idx in range(num_samples):
        # Set seed for this specific sample
        sample_seed = base_seed + sample_idx
        random.seed(sample_seed)
        
        # Select response (with some randomness based on temperature)
        if temperature > 0.5:
            # Higher temperature = more variety
            response = random.choice(all_responses)
        else:
            # Lower temperature = more consistent
            response = random.choice(type2_responses)  # Prefer Type-2
        
        # Create sample
        sample = {
            "prompt_id": prompt_id,
            "text": prompt_text,
            "sample_idx": sample_idx,
            "seed": sample_seed,
            "output_text": response,
            "first_3_token_logits": None,  # Mock doesn't have logits
            "meta": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "model_name": "mock-llama-3-8b-instruct",
                "formatted_prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eos_token|><|start_header_id|>assistant<|end_header_id|>\n\n"
            }
        }
        samples.append(sample)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Mock generation for conflict prompts")
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
                       help="Use deterministic generation")
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
        print("Using deterministic generation")
    
    # Load conflict prompts
    conflict_prompts = load_conflict_prompts()
    print(f"Loaded {len(conflict_prompts)} conflict prompts")
    
    # Generate samples
    all_samples = []
    for prompt in conflict_prompts:
        print(f"Generating samples for {prompt['id']}: {prompt['text'][:50]}...")
        
        samples = mock_generate_samples(
            prompt_text=prompt["text"],
            prompt_id=prompt["id"],
            num_samples=args.samples_per_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            base_seed=args.base_seed
        )
        
        all_samples.extend(samples)
    
    # Save samples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, all_samples)
    
    # Save configuration
    config = {
        "model": {
            "name": "mock-llama-3-8b-instruct",
            "dtype": "bf16",
            "device": "cpu"
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
        "seed_policy": "base_seed + sample_idx for reproducibility",
        "note": "Mock generation for testing - replace with actual model for production"
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
        repro_cmd = f"python src/generate_mock.py --temperature {args.temperature} --top_p {args.top_p} --max_new_tokens {args.max_new_tokens} --samples_per_prompt {args.samples_per_prompt} --base_seed {args.base_seed} --repro_prompt_id {args.repro_prompt_id} --repro_sample_idx {args.repro_sample_idx}"
        print(f"\n🔄 REPRO CMD for {args.repro_prompt_id} sample {args.repro_sample_idx}:")
        print(f"  {repro_cmd}")


if __name__ == "__main__":
    main() 