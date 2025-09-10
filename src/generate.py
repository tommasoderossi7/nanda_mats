#!/usr/bin/env python3
"""
Generate 10 samples per conflict prompt (stochastic) + deterministic eval toggle.
Saves prompt_id, text, sample_idx, seed, output_text, first-3-token logits, meta.
"""

import argparse
import json
import os
import sys
import time
import cProfile
import pstats
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_global_seed, write_jsonl  # kept for compatibility, not used here

# ---- tqdm (optional) ---------------------------------------------------------
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False
    # Minimal no-op shim
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if iterable is None:
            class _Dummy:
                def update(self, n=1): pass
                def close(self): pass
            return _Dummy()
        for i, x in enumerate(iterable, 1):
            if total:
                print(f"{desc or 'progress'}: {i}/{total}", end="\r")
            yield x
        if total:
            print(" " * 60, end="\r")

# ---- Timing / Perf Tracker ---------------------------------------------------
class PerfTracker:
    def __init__(self):
        self.totals = {}    # section -> seconds
        self.counts = {}    # section -> int
        self.counters = {}  # arbitrary counters, e.g., generated_tokens
        self._stack = []    # nested sections allowed

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        self._stack.append(name)
        try:
            yield
        finally:
            end = time.perf_counter()
            dt = end - start
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            self._stack.pop()

    def add_time(self, name: str, seconds: float):
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def incr(self, name: str, value: float = 1.0):
        self.counters[name] = self.counters.get(name, 0.0) + value

    def summary(self) -> Dict:
        total_time = sum(self.totals.values()) or 1e-9
        rows = []
        for name, tot in sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True):
            cnt = self.counts.get(name, 0)
            avg = (tot / max(cnt, 1)) if cnt else 0.0
            pct = 100.0 * tot / total_time
            rows.append({"section": name, "count": cnt, "total_s": tot, "avg_s": avg, "pct": pct})
        return {"total_time_s": total_time, "sections": rows, "counters": dict(self.counters)}

    def print_table(self):
        s = self.summary()
        print("\n⏱️ Timing Summary")
        print(f"  Total wall time: {s['total_time_s']:.3f}s")
        if "generated_tokens" in s["counters"] and self.totals.get("generation"):
            ttoks = int(s["counters"]["generated_tokens"])
            tgen = self.totals["generation"]
            print(f"  Generation throughput: {ttoks} tokens in {tgen:.3f}s  (~{ttoks/max(tgen,1e-9):.1f} toks/s)")
        print("\n  {sec:<22} {cnt:>6}  {tot:>10}  {avg:>10}  {pct:>7}".format(
            sec="Section", cnt="Count", tot="Total(s)", avg="Avg(s)", pct="%"))
        print("  " + "-" * 60)
        for row in s["sections"]:
            print("  {sec:<22} {cnt:>6}  {tot:>10.3f}  {avg:>10.3f}  {pct:>6.1f}%".format(
                sec=row["section"], cnt=row["count"], tot=row["total_s"], avg=row["avg_s"], pct=row["pct"]))
        return s

# ---- Core helpers ------------------------------------------------------------
def load_model_and_tokenizer(
    model_name: str = "./models/llama-3.1-8b-instruct",
    dtype: str = "bf16",
    device: str = "auto",
) -> tuple:
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set dtype (support fp16 properly)
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        local_files_only=True,
    )
    return model, tokenizer


def format_prompt(tokenizer, text: str) -> str:
    """Format prompt using Llama-3.1 chat template."""
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _append_jsonl(handle, obj: dict):
    """Append a JSON line and fsync for durability."""
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    handle.write(line)
    handle.flush()
    os.fsync(handle.fileno())


def generate_and_write_samples(
    model,
    tokenizer,
    prompt_text: str,
    prompt_id: str,
    *,
    num_samples: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 200,
    base_seed: int = 42,
    save_logits: bool = True,
    responses_handle=None,
    logits_handle=None,
    deterministic: bool = False,
    example_holder: dict = None,
    perf: PerfTracker = None,
    show_cuda_mem: bool = False,
    progress_bar: bool = True,
):
    """
    Generate samples for a single prompt and write them progressively.
    Returns: (num_samples_written, num_logits_written)
    """
    # Format prompt
    with perf.section("format_prompt"):
        formatted_prompt = format_prompt(tokenizer, prompt_text)

    # Tokenize input
    with perf.section("tokenize"):
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

    logits_written = 0
    samples_written = 0

    inner_iter = range(num_samples)
    if progress_bar and HAVE_TQDM:
        inner_iter = tqdm(inner_iter, total=num_samples, desc=f"{prompt_id}")

    for sample_idx in inner_iter:
        # Seed policy
        sample_seed = base_seed + sample_idx
        set_global_seed(sample_seed)

        # Generation timing (GPU-aware)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=not deterministic,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=save_logits,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        perf.add_time("generation", t1 - t0)

        # Extract generated text
        with perf.section("decode"):
            generated_ids = outputs.sequences[0]
            gen_len = int(generated_ids.shape[0] - input_ids.shape[1])
            perf.incr("generated_tokens", gen_len)
            generated_text = tokenizer.decode(
                generated_ids[input_ids.shape[1]:], skip_special_tokens=True
            )

        # Build sample record
        sample = {
            "prompt_id": prompt_id,
            "text": prompt_text,
            "sample_idx": sample_idx,
            "seed": sample_seed,
            "output_text": generated_text,
            "meta": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "model_name": model.config._name_or_path,
                "formatted_prompt": formatted_prompt,
                "deterministic": deterministic,
                "generated_tokens": gen_len,
            },
        }

        # Progressive write sample
        if responses_handle is not None:
            with perf.section("write_sample"):
                _append_jsonl(responses_handle, sample)
        samples_written += 1

        # First example for pretty print
        if example_holder is not None and "example" not in example_holder:
            example_holder["example"] = sample

        # Progressive write logits
        if save_logits and hasattr(outputs, "scores") and len(outputs.scores) >= 1:
            first_k = min(3, len(outputs.scores))
            with perf.section("collect_logits"):
                first_k_logits = []
                for i in range(first_k):
                    logits = outputs.scores[i][0].detach().cpu().numpy()
                    first_k_logits.append(logits.tolist())

            logits_entry = {
                "prompt_id": prompt_id,
                "sample_idx": sample_idx,
                "first_3_token_logits": first_k_logits,
            }

            if logits_handle is not None:
                with perf.section("write_logits"):
                    _append_jsonl(logits_handle, logits_entry)
            logits_written += 1

        # Optional CUDA memory stats
        if show_cuda_mem and torch.cuda.is_available() and (sample_idx + 1) % 5 == 0:
            mem = torch.cuda.memory_allocated() / (1024 ** 2)
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if HAVE_TQDM and progress_bar:
                # avoid breaking the bar
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"[{prompt_id}] CUDA mem: {mem:.1f}MB, peak: {max_mem:.1f}MB")
            else:
                print(f"[{prompt_id}] CUDA mem: {mem:.1f}MB, peak: {max_mem:.1f}MB")

    return samples_written, logits_written


def load_conflict_prompts(
    conflict_families: Optional[List[str]] = None,
    specific_prompts: Optional[List[str]] = None,
    prompts_split: str = "dev",
) -> List[Dict]:
    """Load conflict prompts from JSON files with optional filtering."""
    prompts_dir = Path("prompts/separable_constraints")
    all_prompts = []

    if prompts_split == "dev":
        prompt_files = {
            "f1": ["dev_f1_conflicts.json", "dev_f1_nonconf_minpairs.json"],
            #"f2": ["dev_f2_conflicts.json", "dev_f2_nonconf_minpairs.json"],
            #"benign": ["dev_benign.json"],
        }
    else:  # test
        prompt_files = {
            "f1": ["test_f1_conflicts.json", "test_f1_nonconf_minpairs.json"],
            "f2": ["test_f2_conflicts.json", "test_f2_nonconf_minpairs.json"],
            "benign": ["dev_benign.json"],
        }

    families_to_load = conflict_families if conflict_families else ["f1", "f2", "benign"]

    for family in families_to_load:
        if family in prompt_files:
            for filename in prompt_files[family]:
                file_path = prompts_dir / filename
                if file_path.exists():
                    prompts = json.loads(file_path.read_text())
                    all_prompts.extend(prompts)

    if specific_prompts:
        all_prompts = [p for p in all_prompts if p["id"] in specific_prompts]

    return all_prompts


def main_impl(args):
    perf = PerfTracker()

    # Choose output tag
    tag = "exp" if args.prompt_string else args.prompts_to_gen
    responses_output = f"data/{tag}_gens.jsonl"
    logits_output = f"data/{tag}_logits.jsonl"
    config_output = f"data/{tag}_gen_cfg.json"
    timing_output = args.timing_json or f"data/{tag}_timing.json"
    profile_output = args.profile_cpu or f"data/{tag}_cprofile.prof"

    progress_bar = (not args.no_progress_bar)

    # Load model and tokenizer
    with perf.section("load_model_and_tokenizer"):
        model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device)

    # Deterministic note
    if args.deterministic:
        print("Using deterministic (greedy) generation")

    # Filtering args
    conflict_families = [f.strip() for f in args.conflict_family.split(",")] if args.conflict_family else None
    specific_prompts = [p.strip() for p in args.prompts_to_run.split(",")] if args.prompts_to_run else None

    # Load conflict prompts
    with perf.section("load_prompts"):
        if args.prompt_string:
            conflict_prompts = [{"id": "custom_prompt", "text": args.prompt_string}]
        else:
            conflict_prompts = load_conflict_prompts(conflict_families, specific_prompts, args.prompts_to_gen)
    print(f"Loaded {len(conflict_prompts)} prompt(s)")

    # Logits setting
    save_logits = args.save_logits and not args.save_nothing
    print(f"Logits will {'be' if save_logits else 'NOT be'} saved")

    # Prepare output files (progressive append)
    responses_handle = None
    logits_handle = None
    output_path = Path(responses_output)
    logits_path = Path(logits_output)
    config_path = Path(config_output)
    timing_path = Path(timing_output)

    total_samples = 0
    total_logits = 0
    example_holder = {}

    try:
        if not args.save_nothing:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            responses_handle = output_path.open("a", encoding="utf-8")
            if save_logits:
                logits_path.parent.mkdir(parents=True, exist_ok=True)
                logits_handle = logits_path.open("a", encoding="utf-8")

        outer_iter = conflict_prompts
        if progress_bar and HAVE_TQDM:
            outer_iter = tqdm(conflict_prompts, total=len(conflict_prompts), desc="Prompts")

        for prompt in outer_iter:
            # Generate per prompt
            with perf.section("per_prompt"):
                n_s, n_l = generate_and_write_samples(
                    model,
                    tokenizer,
                    prompt_text=prompt["text"],
                    prompt_id=prompt["id"],
                    num_samples=args.samples_per_prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    base_seed=args.base_seed,
                    save_logits=save_logits,
                    responses_handle=responses_handle,
                    logits_handle=logits_handle,
                    deterministic=args.deterministic,
                    example_holder=example_holder,
                    perf=perf,
                    show_cuda_mem=args.show_cuda_mem,
                    progress_bar=progress_bar,
                )
            total_samples += n_s
            total_logits += n_l

    finally:
        if responses_handle is not None:
            responses_handle.close()
        if logits_handle is not None:
            logits_handle.close()

    # Save configuration
    config = {
        "model": {"name": args.model, "dtype": args.dtype, "device": args.device},
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "deterministic": args.deterministic,
        },
        "generation": {
            "samples_per_prompt": args.samples_per_prompt,
            "base_seed": args.base_seed,
            "total_prompts": len(conflict_prompts),
            "total_samples": total_samples,
            "save_logits": save_logits,
            "total_logits": total_logits,
        },
        "seed_policy": "base_seed + sample_idx for reproducibility",
        "outputs": {"responses": str(output_path), "logits": str(logits_path) if save_logits else None},
    }

    if args.save_nothing:
        print("Not saving config as per --save_nothing flag.")
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    # Print timing/table and save JSON
    timing_summary = perf.print_table()
    try:
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_path.open("w", encoding="utf-8") as f:
            json.dump(timing_summary, f, indent=2)
        print(f"\n🕒 Timing JSON saved to: {timing_path}")
    except Exception as e:
        print(f"Could not write timing JSON: {e}")

    # Summary
    print(f"\n📊 Generation Summary:")
    print(f"  Total prompts: {len(conflict_prompts)}")
    print(f"  Samples per prompt: {args.samples_per_prompt}")
    print(f"  Total samples: {total_samples}")
    if not args.save_nothing:
        print(f"  Output (appended): {output_path}")
        if save_logits:
            print(f"  Logits (appended): {logits_path}")
        print(f"  Config: {config_path}")

    # Example row
    if "example" in example_holder:
        example = example_holder["example"]
        print(f"\n📝 Example row:")
        print(f"  prompt_id: {example['prompt_id']}")
        print(f"  sample_idx: {example['sample_idx']}")
        print(f"  text: {example['text'][:50]}...")
        print(f"  output_text: {example['output_text'][:50]}...")

    # Optional CUDA memory wrap-up
    if args.show_cuda_mem and torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024 ** 2)
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n🧠 CUDA memory (end): {mem:.1f}MB, peak: {max_mem:.1f}MB")

def main():
    parser = argparse.ArgumentParser(description="Generate samples for conflict prompts (with progress & timing)")
    parser.add_argument("--model", default="./models/llama-3.1-8b-instruct", help="Model name or path")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"], help="Model dtype")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum new tokens")
    parser.add_argument("--samples_per_prompt", type=int, default=10, help="Number of samples per prompt")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic (greedy) generation")
    parser.add_argument("--prompts_to_gen", default="dev", help="generations split to label (dev/test)")
    parser.add_argument("--save_logits", action="store_true", default=False, help="Save first 3 token logits")
    parser.add_argument("--save_nothing", action="store_true", help="Don't save anything (for quick tests)")
    parser.add_argument("--repro_prompt_id", help="Prompt ID for REPRO CMD")
    parser.add_argument("--repro_sample_idx", type=int, help="Sample index for REPRO CMD")
    parser.add_argument("--conflict_family", help="Comma-separated list of conflict families to run (e.g., f1,f2)")
    parser.add_argument("--prompts_to_run", help="Comma-separated list of specific prompt IDs to run (e.g., f1_002_nonconf,f2_001)")
    parser.add_argument("--prompt_string", type=str, help="A custom prompt string to generate samples for")
    #parser.add_argument("--prompts_type", type=str, help="Whether to use separable constraints prompt or not", choices=["separable", "nonseparable"], default="separable")

    # New flags
    parser.add_argument("--no_progress_bar", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--timing_json", type=str, help="Path to save timing summary JSON (default auto)")
    parser.add_argument("--profile_cpu", type=str, help="Write a cProfile .prof file (default auto path)")
    parser.add_argument("--show_cuda_mem", action="store_true", help="Periodically show CUDA memory usage")

    args = parser.parse_args()

    # Optional CPU profiler
    if args.profile_cpu is not None:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            main_impl(args)
        finally:
            profiler.disable()
            # Default path selection if given flag used without explicit path
            tag = "exp" if args.prompt_string else args.prompts_to_gen
            out = args.profile_cpu or f"data/{tag}_cprofile.prof"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(out)
            print(f"\n🧩 cProfile stats saved to: {out}")
            print("  Hint: use `snakeviz {}` or `gprof2dot -f pstats {} | dot -Tpng -o prof.png`".format(out, out))
    else:
        main_impl(args)


if __name__ == "__main__":
    main()