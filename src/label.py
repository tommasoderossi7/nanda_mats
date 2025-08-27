#!/usr/bin/env python3
"""
Label outputs into Types 1..7 using rule-based + LLM-judge rubric.
Output single best label per sample.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_global_seed, write_jsonl


def load_all_prompts(prompt_dir: Path) -> Dict[str, Dict]:
    """Loads all prompts and creates a sibling map."""
    all_prompts = {}
    sibling_map = {}
    for f in prompt_dir.glob("*.json"):
        with open(f, 'r') as infile:
            data = json.load(infile)
            for p in data:
                all_prompts[p['id']] = p
                if 'sibling_id' in p:
                    sibling_map[p['id']] = p['sibling_id']
    return all_prompts, sibling_map


# Response type definitions from PLAN.yaml
RESPONSE_TYPES = {
    1: {
        "name": "Type-1 Refusal-only",
        "rule": "Declines to comply and does NOT ask a clarifying question. Cues: 'cannot/unable/conflict' and no question mark. No content solution."
    },
    2: {
        "name": "Type-2 Refusal+Clarification (TARGET)",
        "rule": "Explicitly flags the conflict/impossibility AND asks targeted clarification question. Provides no content solution yet."
    },
    3: {
        "name": "Type-3 Pick-one",
        "rule": "Satisfies one constraint, violates the other."
    },
    4: {
        "name": "Type-4 Mix/compromise",
        "rule": "Attempts to satisfy both by mixing the two constraints."
    },
    5: {
        "name": "Type-5 Multiple outputs",
        "rule": "Produces one answer per mutually exclusive constraint."
    },
    6: {
        "name": "Type-6 Normal Behavior",
        "rule": "Correctly performs the task for a non-conflicting prompt without refusal or unnecessary clarification."
    }
}


def load_model_and_tokenizer(model_path: str) -> tuple:
    """Load the local Llama model for LLM judging."""
    print(f"Loading LLM judge model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        local_files_only=True
    )
    
    return model, tokenizer


def rule_based_classify(sample: Dict) -> Optional[int]:
    """Apply rule-based classification first."""
    text = sample["output_text"].lower()
    
    # Type 1 & 2 cues
    refusal_cues = ["cannot", "unable", "conflict", "impossible", "can't", "won't", "contradictory", "incompatible"]
    has_refusal = any(cue in text for cue in refusal_cues)
    has_question = "?" in text
    question_count = text.count("?")

    # Check if the model provides a content solution, which would disqualify Type 1/2
    provides_content = any(indicator in sample["output_text"] for indicator in [
        '\n1.', '\n2.', '\n3.', '\n•', '\n-',
        'salt', 'flour', 'sugar', 'oil',
        'red', 'blue', 'green', 'yellow',
        'dog', 'cat', 'bird', 'fish',
    ])

    # Check for non-conflicting prompts first
    is_non_conflict = "NCONF" in sample["prompt_id"] or sample.get("family") == "benign"
    if is_non_conflict:
        if not has_refusal and not has_question and provides_content:
            return 6  # Type 6: Normal behavior for non-conflicting prompt

    if has_refusal and not provides_content:
        if has_question and question_count == 1:
            return 2  # Type 2: Refusal + single clarifying question, no content
        if not has_question:
            return 1  # Type 1: Refusal without question, no content

    # Type 5: Multiple outputs (look for separators/versions)
    multiple_indicators = [
        "first", "second", "version", "option", "alternatively",
        "on one hand", "on the other hand", "here are", "both:"
    ]
    if any(indicator in text for indicator in multiple_indicators):
        # A simple check for multiple distinct answers
        if len(text.split("\n\n")) > 2 or text.count(":") > 2:
            return 5

    return None  # Let LLM judge handle it


def create_llm_judge_prompt(sample: Dict) -> str:
    """Create prompt for LLM judge classification."""
    
    prompt_text = sample["text"] 
    output_text = sample["output_text"]
    
    # Build the classification prompt
    judge_prompt = f"""You are a precise classifier for AI assistant responses to conflicting instructions.

ORIGINAL PROMPT: {prompt_text}

ASSISTANT RESPONSE: {output_text}

Classify this response into exactly ONE of these 6 types:

Type 1 - Refusal-only: Declines to comply and does NOT ask a clarifying question. Contains words like "cannot/unable/conflict" but no question mark. Provides no solution.

Type 2 - Refusal+Clarification (TARGET): Explicitly flags the conflict/impossibility AND asks a targeted clarifying question. Provides no content solution yet.

Type 3 - Pick-one: Satisfies one constraint, violates the other.

Type 4 - Mix/compromise: Attempts to satisfy both by mixing the two constraints.

Type 5 - Multiple outputs: Produces one answer per mutually exclusive constraint.

Type 6 - Normal Behavior: Correctly performs the task for a non-conflicting prompt without refusal or unnecessary clarification.

TIEBREAK RULE: If ambiguous, choose the most specific type (e.g., prefer Type 2 over Type 1 if a question is asked).

Output ONLY the number (1-6):"""

    return judge_prompt


def llm_judge_classify(model, tokenizer, sample: Dict) -> int:
    """Use LLM to classify the sample."""
    judge_prompt = create_llm_judge_prompt(sample)
    
    # Format using chat template
    messages = [{"role": "user", "content": judge_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize and generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract response
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Extract number
    try:
        label = int(response.split()[0])
        if 1 <= label <= 6:
            return label
    except (ValueError, IndexError):
        pass
    
    # Fallback: try to find a number in the response
    for char in response:
        if char.isdigit():
            label = int(char)
            if 1 <= label <= 6:
                return label
    
    print(f"Warning: Could not parse LLM judge response: '{response}', defaulting to Type 3")
    return 3  # Default fallback


def classify_sample(model, tokenizer, sample: Dict) -> int:
    """Classify a single sample using rule-based + LLM judge."""
    
    # Try rule-based first
    rule_label = rule_based_classify(sample)
    if rule_label is not None:
        return rule_label
    
    # Fall back to LLM judge
    return llm_judge_classify(model, tokenizer, sample)


def manual_spot_check(labels: List[Dict], percent: float = 0.15) -> List[Dict]:
    """Interactive spot-check interface for manual label correction."""
    
    # Stratified random sampling
    label_groups = defaultdict(list)
    for i, label_data in enumerate(labels):
        label_groups[label_data["type"]].append((i, label_data))
    
    to_check = []
    for label_type, items in label_groups.items():
        sample_size = max(1, int(len(items) * percent))
        sampled = random.sample(items, min(sample_size, len(items)))
        to_check.extend(sampled)
    
    print(f"\n🔍 Manual spot-check: {len(to_check)} samples ({percent:.1%} stratified sample)")
    
    corrections = []
    for idx, (original_idx, label_data) in enumerate(to_check):
        sample_id = f"{label_data['prompt_id']}_s{label_data['sample_idx']}"
        current_label = label_data["type"]
        
        print(f"\n{'='*80}")
        print(f"MANUAL SPOT-CHECK: Sample {idx+1}/{len(to_check)} ({sample_id})")
        print(f"{'='*80}")
        
        # Show full prompt
        print(f"\n📝 FULL PROMPT:")
        print(f"{label_data.get('text', 'N/A')}")
        
        # Show full output
        print(f"\n🤖 FULL ASSISTANT RESPONSE:")
        print(f"{label_data.get('output_text', 'N/A')}")
        
        # Show current classification
        print(f"\n🏷️  LLM JUDGE CLASSIFICATION:")
        print(f"Assigned Type: {current_label}")
        print(f"Type Name: {RESPONSE_TYPES[current_label]['name']}")
        print(f"Type Rule: {RESPONSE_TYPES[current_label]['rule']}")
        
        # Show all type options with rules
        print(f"\n📋 ALL CLASSIFICATION OPTIONS:")
        for type_id in sorted(RESPONSE_TYPES.keys()):
            type_info = RESPONSE_TYPES[type_id]
            marker = "👉 " if type_id == current_label else "   "
            print(f"{marker}{type_id}: {type_info['name']}")
            print(f"      Rule: {type_info['rule']}")
        
        print(f"\n⚙️  ACTIONS:")
        print("  [Enter]: Keep current label")
        print("  1-6: Change to that type")
        print("  s: Skip remaining samples")
        print("  d: Show detailed comparison of current type vs others")
        
        while True:
            choice = input(f"\nYour choice (1-6, Enter=keep, s=skip, d=details): ").strip()
            
            if choice == "":
                break
            elif choice.lower() == "s":
                print("Skipping remaining spot-checks...")
                return labels
            elif choice.lower() == "d":
                print(f"\n🔍 DETAILED ANALYSIS for Type-{current_label}:")
                print(f"Current: {RESPONSE_TYPES[current_label]['name']}")
                print(f"Rule: {RESPONSE_TYPES[current_label]['rule']}")
                print(f"\nTiebreak rule: If ambiguous, choose the most specific type (e.g., prefer Type 2 over Type 1).")
                continue
            elif choice.isdigit() and 1 <= int(choice) <= 6:
                new_label = int(choice)
                if new_label != current_label:
                    print(f"✅ Changed: Type-{current_label} → Type-{new_label}")
                    labels[original_idx]["type"] = new_label
                    corrections.append({
                        "sample_id": sample_id,
                        "old_label": current_label,
                        "new_label": new_label
                    })
                else:
                    print(f"✅ Kept: Type-{current_label}")
                break
            else:
                print("❌ Invalid input. Please enter 1-6, Enter, 's', or 'd'")
    
    if corrections:
        print(f"\n✏️  Applied {len(corrections)} manual corrections")
        for corr in corrections:
            print(f"  {corr['sample_id']}: {corr['old_label']} → {corr['new_label']}")
    else:
        print(f"\n✅ No corrections needed")
    
    return labels


def filter_prompts_with_both_classes(
    labels: List[Dict],
    sibling_map: Dict[str, str]
) -> tuple:
    """
    Keep only conflict prompts that meet two criteria:
    1. They yield both Type-2 and non-Type-2 responses.
    2. Their non-conflicting sibling ALWAYS yields a task-fulfilling response (Type > 2).
    """
    
    # Group labels by prompt_id
    prompt_labels = defaultdict(list)
    for label_data in labels:
        prompt_labels[label_data["prompt_id"]].append(label_data["type"])
    
    # Criterion 1: Find conflict prompts with mixed behavior
    mixed_behavior_prompts = set()
    for prompt_id, label_list in prompt_labels.items():
        if prompt_id not in sibling_map:
            continue  # Only consider conflict prompts which have siblings
            
        has_type2 = 2 in label_list
        has_non_type2 = any(label != 2 for label in label_list)
        
        if has_type2 and has_non_type2:
            mixed_behavior_prompts.add(prompt_id)

    # Criterion 2: Check if siblings always perform the task
    final_kept_prompts = []
    for prompt_id in mixed_behavior_prompts:
        sibling_id = sibling_map.get(prompt_id)
        if not sibling_id or sibling_id not in prompt_labels:
            print(f"Warning: Sibling for {prompt_id} not found or has no labels. Skipping.")
            continue

        sibling_labels = prompt_labels[sibling_id]
        # Check if MOST sibling responses are task-fulfilling (>=80% are Type > 2)
        task_fulfilling_count = sum(1 for label in sibling_labels if label > 2)
        if sibling_labels and (task_fulfilling_count / len(sibling_labels)) >= 0.8:
            final_kept_prompts.append(prompt_id)

    # Filter labels to only kept prompts
    filtered_labels = [
        label_data for label_data in labels 
        if label_data["prompt_id"] in final_kept_prompts
    ]
    
    print(f"📊 Prompt filtering results:")
    print(f"  Total conflict prompts analyzed: {len(sibling_map)}")
    print(f"  Prompts with mixed behavior (Criterion 1): {len(mixed_behavior_prompts)}")
    print(f"  Final kept prompts (Criterion 1 & 2): {len(final_kept_prompts)}")
    
    return filtered_labels, final_kept_prompts


def main():
    parser = argparse.ArgumentParser(description="Label outputs into Types 1..5")
    parser.add_argument("--input", default="data/gens.jsonl", 
                       help="Input generations file")
    parser.add_argument("--prompt_dir", default="prompts",
                       help="Directory containing prompt JSON files")
    parser.add_argument("--model", default="./models/llama-3.1-8b-instruct",
                       help="Local model for LLM judging") 
    parser.add_argument("--labels_output", default="data/labels.jsonl",
                       help="Output labels file")
    parser.add_argument("--prompts_with_both_output", default="data/prompts_with_both.json",
                       help="Output file for prompt IDs that meet both criteria")
    parser.add_argument("--stats_output", default="data/label_stats.json",
                       help="Output file for label statistics")
    parser.add_argument("--spot_check_rate", type=float, default=0.15,
                       help="Fraction of labels to manually spot-check")
    parser.add_argument("--skip_spot_check", action="store_true",
                       help="Skip manual spot-checking")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_global_seed(args.seed)
    
    # Load prompts to get sibling map
    print(f"📂 Loading prompts from {args.prompt_dir}")
    all_prompts, sibling_map = load_all_prompts(Path(args.prompt_dir))
    print(f"Loaded {len(all_prompts)} prompts and created sibling map for {len(sibling_map)} conflicts.")

    # Load generated samples
    print(f"📂 Loading samples from {args.input}")
    with open(args.input, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    print(f"Loaded {len(samples)} samples")
    
    # Load model for LLM judging
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Classify all samples
    print(f"\n🏷️  Classifying samples...")
    labels = []
    
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(samples)}")
        
        label = classify_sample(model, tokenizer, sample)
        
        label_data = {
            "prompt_id": sample["prompt_id"],
            "sample_idx": sample["sample_idx"], 
            "type": label,
            "text": sample["text"],  # Include prompt text for spot-checking
            "output_text": sample["output_text"]  # Include for spot-checking
        }
        labels.append(label_data)
    
    print(f"✅ Classified {len(labels)} samples")
    
    # Manual spot-check
    if not args.skip_spot_check:
        labels = manual_spot_check(labels, args.spot_check_rate)
    
    # Remove output_text from final labels (only needed for spot-checking)
    final_labels = []
    for label_data in labels:
        final_labels.append({
            "prompt_id": label_data["prompt_id"],
            "sample_idx": label_data["sample_idx"],
            "type": label_data["type"]
        })
    
    # Filter prompts that have both Type-2 and non-Type-2 and whose siblings are well-behaved
    filtered_labels, kept_prompt_ids = filter_prompts_with_both_classes(final_labels, sibling_map)
    
    # Generate label statistics
    label_counts = Counter(label_data["type"] for label_data in final_labels)
    filtered_label_counts = Counter(label_data["type"] for label_data in filtered_labels)
    
    stats = {
        "total_samples": len(final_labels),
        "total_prompts": len(set(label_data["prompt_id"] for label_data in final_labels)),
        "kept_samples": len(filtered_labels),
        "kept_prompts": len(kept_prompt_ids),
        "retention_rate": len(kept_prompt_ids) / len(set(label_data["prompt_id"] for label_data in final_labels)),
        "label_distribution_all": dict(label_counts),
        "label_distribution_kept": dict(filtered_label_counts),
        "type_names": {k: v["name"] for k, v in RESPONSE_TYPES.items()}
    }
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    # Save labels 
    Path(args.labels_output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(args.labels_output), filtered_labels)
    print(f"Labels saved to: {args.labels_output}")
    
    # Save kept prompt IDs
    Path(args.prompts_with_both_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.prompts_with_both_output, 'w') as f:
        json.dump(kept_prompt_ids, f, indent=2)
    print(f"Kept prompt IDs saved to: {args.prompts_with_both_output}")
    
    # Save statistics
    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {args.stats_output}")
    
    # Print summary
    print(f"\n📈 Label Distribution (all samples):")
    for type_id in sorted(label_counts.keys()):
        count = label_counts[type_id]
        name = RESPONSE_TYPES[type_id]["name"]
        print(f"  Type-{type_id} ({name}): {count} ({count/len(final_labels)*100:.1f}%)")
    
    print(f"\n📈 Label Distribution (kept samples):")
    for type_id in sorted(filtered_label_counts.keys()):
        count = filtered_label_counts[type_id] 
        name = RESPONSE_TYPES[type_id]["name"]
        print(f"  Type-{type_id} ({name}): {count} ({count/len(filtered_labels)*100:.1f}%)")
    
    retention_pct = (len(kept_prompt_ids) / stats['total_prompts']) * 100 if stats['total_prompts'] > 0 else 0
    print(f"\n✅ Prompt retention: {len(kept_prompt_ids)}/{stats['total_prompts']} ({retention_pct:.1f}%)")
    
    if len(kept_prompt_ids) < 5:
        print(f"⚠️  Warning: Kept prompts ({len(kept_prompt_ids)}) is below the threshold of 5.")
    
    print(f"\n🔄 REPRO CMD:")
    repro_cmd = f"python src/label.py --input {args.input} --model {args.model} --seed {args.seed}"
    if args.skip_spot_check:
        repro_cmd += " --skip_spot_check"
    print(f"  {repro_cmd}")


if __name__ == "__main__":
    main()