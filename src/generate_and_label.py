#!/usr/bin/env python3
"""
Run generation then labeling in sequence.

Examples:
  python src/run_pipeline.py \
    --prompts_to_gen dev \
    --gen_model ./models/llama-3.1-8b-instruct \
    --openrouter_model deepseek/deepseek-chat-v3.1:free \
    --samples_per_prompt 5 \
    --save_logits \
    --save_output \
    --skip_spot_check

For a custom one-off prompt:
  python src/run_pipeline.py \
    --prompt_string "Write a haiku about conflicting rules" \
    --gen_model ./models/llama-3.1-8b-instruct \
    --openrouter_model deepseek/deepseek-chat-v3.1:free \
    --save_output
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Generate then label")
    # Choose dataset OR custom prompt
    parser.add_argument("--prompts_to_gen", default="dev",
                        help="Label for generation split (dev/test). Ignored if --prompt_string used.")
    parser.add_argument("--prompt_string", type=str,
                        help="Custom single prompt; sets tag=exp so labeler reads data/exp_*.jsonl")

    # Generation knobs (subset; add more as needed)
    parser.add_argument("--gen_model", default="./models/llama-3.1-8b-instruct")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--samples_per_prompt", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save_logits", action="store_true")

    # Labeling knobs
    parser.add_argument("--openrouter_model", default="deepseek/deepseek-chat-v3.1:free")
    parser.add_argument("--responses_to_label", default="all")
    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--skip_spot_check", action="store_true")
    parser.add_argument("--spot_check_rate", type=float, default=0.15)

    # passthroughs / QoL
    parser.add_argument("--no_progress_bar", action="store_true", help="Hide tqdm in generation")
    parser.add_argument("--show_cuda_mem", action="store_true", help="Show CUDA mem snapshots during generation")

    args = parser.parse_args()

    # Determine tag used by files
    tag = "exp" if args.prompt_string else args.prompts_to_gen

    # 1) GENERATION
    gen_cmd = [
        sys.executable, "src/generate.py",
        "--model", args.gen_model,
        "--dtype", args.dtype,
        "--device", args.device,
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--max_new_tokens", str(args.max_new_tokens),
        "--samples_per_prompt", str(args.samples_per_prompt),
        "--base_seed", str(args.base_seed),
        "--prompts_to_gen", args.prompts_to_gen,
    ]

    if args.prompt_string:
        gen_cmd += ["--prompt_string", args.prompt_string]
    if args.deterministic:
        gen_cmd += ["--deterministic"]
    if args.save_logits:
        gen_cmd += ["--save_logits"]
    if args.no_progress_bar:
        gen_cmd += ["--no_progress_bar"]
    if args.show_cuda_mem:
        gen_cmd += ["--show_cuda_mem"]

    print("🚀 Running generation:")
    print(" ", " ".join(gen_cmd))
    subprocess.run(gen_cmd, check=True)

    # 2) LABELING (reads data/{tag}_gens.jsonl produced above)
    label_cmd = [
        sys.executable, "src/label.py",
        "--gens_to_label", tag,
        "--model", args.openrouter_model,
        "--responses_to_label", str(args.responses_to_label),
    ]
    if args.save_output:
        label_cmd += ["--save_output"]
    if args.skip_spot_check:
        label_cmd += ["--skip_spot_check"]
    else:
        label_cmd += ["--spot_check_rate", str(args.spot_check_rate)]

    print("\n🏷️ Running labeling:")
    print(" ", " ".join(label_cmd))
    # Make sure OpenRouter key is available for the subprocess
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No API key in environment (.env not found or missing). Labeling may fail.", file=sys.stderr)

    subprocess.run(label_cmd, check=True)

    print("\n✅ Pipeline complete.")
    print(f"   Generated: data/{tag}_gens.jsonl")
    print(f"   Raw labels (progressive): data/{tag}_labels_raw.jsonl (if --save_output)")
    print(f"   Final labels: data/{tag}_labels.jsonl (if --save_output)")
    print(f"   Stats: data/{tag}_label_stats.json (if --save_output)")

if __name__ == "__main__":
    main()