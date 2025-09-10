#!/usr/bin/env python3
"""
Extract residual activations for steering vector computation (ACROSS plan).

Plan step: 4-acts-across
- Teacher-force the first 5 assistant tokens and capture residual stream activations
  at post-instruction sites (the first 5 assistant tokens) for conflict_train samples.
- Map “mid/late” to concrete layer indices (depth-aware).
- Save arrays per (layer, timestep) with [n, d_model] and full (prompt_id, sample_idx) alignment.

Outputs:
  data/acts/across/train.npz
    keys: layer_mid_t1, layer_mid_t2, layer_mid_t3, layer_mid_t4, layer_mid_t5, layer_late_t1, layer_late_t2, layer_late_t3, layer_late_t4, layer_late_t5
    values: numpy arrays [n, d_model] with the SAME n for all keys
  data/acts/across/train_meta.json
    { layer_indices, timesteps, n, d_model, model_path, samples:[{prompt_id,sample_idx,type,aggregate}] }
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Make `src` importable when running from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import set_global_seed, read_jsonl  # type: ignore


# ---------------------- helpers ----------------------

def depth_aware_indices(num_layers: int) -> Dict[str, int]:
    """
    Map symbolic names to concrete indices (≈0.4D and ≈0.8D), clamped to [0, D-1],
    and ensure distinct indices when possible.
    """
    if num_layers <= 0:
        raise ValueError("Model reports zero layers.")

    mid = int(round(0.4 * (num_layers - 1)))
    late = int(round(0.8 * (num_layers - 1)))

    mid = max(0, min(num_layers - 1, mid))
    late = max(0, min(num_layers - 1, late))

    # ensure distinct if they collapsed (can happen on tiny models)
    if late == mid and num_layers > 1:
        late = min(num_layers - 1, mid + 1)

    return {"mid": mid, "late": late}


def load_model_and_tokenizer(model_path: str, device: str = "auto") -> Tuple[Any, Any]:
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded; dtype={model.dtype}, device(s) set via device_map={device}")
    return model, tokenizer


def ensure_three_tokens(tokenizer: Any, text: str, min_len: int = 5) -> bool:
    """Return True if tokenized 'text' has at least min_len tokens (no special tokens)."""
    toks = tokenizer.encode(text or "", add_special_tokens=False)
    return len(toks) >= min_len


def teacher_force_hidden_states(
    model: Any,
    tokenizer: Any,
    formatted_prompt: str,
    target_text: str,
    layer_indices: Dict[str, int],
    timesteps: List[int]
) -> Dict[str, np.ndarray]:
    """
    Teacher-force the first t tokens (t in timesteps) of target_text,
    run with output_hidden_states=True, and extract the hidden state at
    the last position for each requested layer site.

    Returns dict: keys like 'layer_mid_t1' : np.ndarray [d_model]
    """
    out: Dict[str, np.ndarray] = {}

    # Tokenize prompt and target (no special tokens on target)
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
    target_ids_full = tokenizer.encode(target_text, add_special_tokens=False)

    # Move to model device
    input_ids = input_ids.to(next(model.parameters()).device)

    with torch.no_grad():
        for t in timesteps:
            # Slice the first t tokens from target
            t_slice = target_ids_full[:t]
            if len(t_slice) != t:
                # Shouldn't happen if caller filtered by min length
                continue

            teacher_forced_ids = torch.cat(
                [input_ids, torch.tensor([t_slice], device=input_ids.device)],
                dim=1
            )

            outputs = model(
                teacher_forced_ids,
                output_hidden_states=True,
                use_cache=False
            )

            # outputs.hidden_states: tuple length L+1 (embeddings + per-layer outputs)
            hidden_states = outputs.hidden_states  # tuple of tensors [bsz, seq, d_model]
            target_pos = teacher_forced_ids.shape[1] - 1  # last token position

            for site_name, layer_idx in layer_indices.items():
                # layer output lives at hidden_states[layer_idx+1]
                hs_layer = hidden_states[layer_idx + 1][0, target_pos, :]  # [d_model]
                out[f"layer_{site_name}_t{t}"] = hs_layer.detach().float().cpu().numpy()

    return out


def load_conflict_train_ids(path: Path) -> List[str]:
    """
    Load conflict_train split. Support two formats:
      - [{"prompt_id": "X"}, ...]
      - ["X", "Y", ...]
    Return list of prompt_ids.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    ids: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict) and "prompt_id" in item:
                ids.append(item["prompt_id"])
            else:
                # best-effort: if dict with 'id'
                if isinstance(item, dict) and "id" in item:
                    ids.append(item["id"])
    else:
        raise ValueError(f"Unexpected format in {path}")
    return ids


# ---------------------- main ----------------------

def main():
    parser = argparse.ArgumentParser(description="Extract activations (ACROSS) for conflict_train")
    parser.add_argument("--model_path", default="./models/llama-3.1-8b-instruct", help="HF model path")
    parser.add_argument("--device", default="auto", help="Device map for model (e.g., 'auto')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gens_path", default="data/dev_gens.jsonl", help="Generations JSONL")
    parser.add_argument("--labels_path", default="data/dev_labels_corrected.jsonl", help="Labels JSONL")
    parser.add_argument("--train_split", default="data/splits/conflict_train.json", help="conflict_train split file")
    parser.add_argument("--output_dir", default="data/acts/across", help="Output directory (will create)")
    args = parser.parse_args()

    set_global_seed(args.seed)

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    num_layers = len(model.model.layers)  # type: ignore[attr-defined]
    layer_map = depth_aware_indices(num_layers)
    print(f"Model depth={num_layers}; layer_map={layer_map}")

    # Timesteps: first 3 assistant tokens
    timesteps = [1, 2, 3, 4, 5]

    # Load conflict_train IDs
    train_ids = load_conflict_train_ids(Path(args.train_split))
    train_id_set = set(train_ids)
    print(f"Loaded conflict_train IDs: {len(train_ids)}")

    # Load gens and labels
    gens = read_jsonl(args.gens_path)
    labels = read_jsonl(args.labels_path)

    # Build lookups
    gen_by_key = {(g["prompt_id"], g["sample_idx"]): g for g in gens}
    lab_by_key = {(l["prompt_id"], l["sample_idx"]): l for l in labels}

    # Gather candidate samples for conflict_train only
    # And ensure the response has >=5 tokens (so all keys share the same n)
    samples: List[Dict[str, Any]] = []
    for (pid, sidx), g in gen_by_key.items():
        if pid not in train_id_set:
            continue
        lab = lab_by_key.get((pid, sidx))
        if lab is None:
            continue
        # Ensure we have 5 tokens to teacher-force
        if not ensure_three_tokens(tokenizer, g.get("output_text", ""), min_len=5):
            continue
        samples.append({
            "prompt_id": pid,
            "sample_idx": sidx,
            "gen": g,
            "lab": lab
        })

    # Sort for deterministic order (by prompt_id, then sample_idx)
    samples.sort(key=lambda x: (x["prompt_id"], x["sample_idx"]))
    print(f"Eligible training samples (>=5 tokens): {len(samples)}")

    if len(samples) == 0:
        raise RuntimeError("No samples with >= 5 tokens found in conflict_train; cannot proceed.")

    # Extract activations
    all_keys = [f"layer_{site}_t{t}" for site in ("mid", "late") for t in timesteps]
    buckets: Dict[str, List[np.ndarray]] = {k: [] for k in all_keys}
    meta_samples: List[Dict[str, Any]] = []

    for i, s in enumerate(samples, 1):
        if i % 25 == 0 or i == 1:
            print(f"Processing {i}/{len(samples)}")

        formatted_prompt = s["gen"]["meta"]["formatted_prompt"]
        output_text = s["gen"]["output_text"]

        acts = teacher_force_hidden_states(
            model=model,
            tokenizer=tokenizer,
            formatted_prompt=formatted_prompt,
            target_text=output_text,
            layer_indices=layer_map,
            timesteps=timesteps
        )

        # Sanity: ensure we got all expected keys for this sample
        if not all(k in acts for k in all_keys):
            # If something is missing (shouldn't happen due to prefilter), skip this sample fully
            print(f"  ⚠️ Missing activations for sample ({s['prompt_id']}, {s['sample_idx']}); skipping.")
            continue

        for k in all_keys:
            buckets[k].append(acts[k])

        meta_samples.append({
            "prompt_id": s["prompt_id"],
            "sample_idx": s["sample_idx"],
            "type": s["lab"].get("type"),
            "aggregate": s["lab"].get("aggregate")
        })

    # Convert to arrays and verify equal n
    final: Dict[str, np.ndarray] = {}
    n_ref = None
    d_model = None

    print("Converting to numpy arrays...")
    for k in all_keys:
        arr = np.stack(buckets[k], axis=0) if len(buckets[k]) > 0 else None
        if arr is None:
            raise RuntimeError(f"No activations collected for key {k}.")
        final[k] = arr  # [n, d_model]
        print(f"  {k}: shape {arr.shape}")
        if n_ref is None:
            n_ref = arr.shape[0]
            d_model = arr.shape[1]
        else:
            if arr.shape[0] != n_ref:
                raise RuntimeError(f"Count mismatch for {k}: got n={arr.shape[0]} vs ref n={n_ref}")
            if arr.shape[1] != d_model:
                raise RuntimeError(f"d_model mismatch for {k}: got {arr.shape[1]} vs ref {d_model}")

    # Meta & save
    assert n_ref is not None and d_model is not None
    meta = {
        "layer_indices": layer_map,
        "timesteps": timesteps,
        "n": n_ref,
        "d_model": d_model,
        "model_path": args.model_path,
        "samples": meta_samples
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acts_path = out_dir / "train.npz"
    meta_path = out_dir / "train_meta.json"

    print(f"Saving activations to: {acts_path}")
    np.savez_compressed(acts_path, **final)

    print(f"Saving metadata to: {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Acceptance-style logging
    print("\n=== SUMMARY ===")
    print(f"n (rows per key): {n_ref}")
    print(f"d_model: {d_model}")
    print(f"layer_indices: {layer_map}")
    print(f"keys: {list(final.keys())}")

    # Alignment verification
    prompt_ids_in_meta = [s["prompt_id"] for s in meta_samples]
    unique_prompts = len(set(prompt_ids_in_meta))
    print(f"Alignment OK. Rows={len(meta_samples)} across {unique_prompts} prompts in conflict_train.")

    print(f"\nREPRO CMD:")
    print(f"python src/acts.py --model_path {args.model_path} --device {args.device} --seed {args.seed} "
          f"--gens_path {args.gens_path} --labels_path {args.labels_path} "
          f"--train_split {args.train_split} --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
