#!/usr/bin/env python3
"""
Step 5-vector-multi-across (Updated for stochastic validation and bfloat16 fix)

Build multi-answers (Type-5) direction v* via candidate grid + validation selection.

Inputs:
  - data/acts/across/train.npz
      keys: layer_mid_t1..layer_late_t5
      values: [n, d_model]
  - data/acts/across/train_meta.json
      {
        "layer_indices": {"mid": int, "late": int},
        "timesteps": [1,2,3,4,5],
        "n": int,
        "d_model": int,
        "samples": [{"prompt_id", "sample_idx", "type", "aggregate"}, ...]
      }
  - data/splits/conflict_validation.json -> [{"prompt_id": "..."}]
  - data/splits/controls_gold.json       -> [{"prompt_id": "..."}]
  - prompts/*.json                       -> resolve prompt_id -> text

Outputs:
  - artifacts/v_star_across.npz
      { vector: [d_model], layer_name, layer_index, pos, alpha, delta_magnitude, timesteps }
  - artifacts/selection_table_across.json
      [{ layer_name, layer_index, pos, alpha, delta_Type5, delta_not_Type5, mean_KL_controls,
         Top10_overlap_controls, J_multi, n_val, n_ctrl }, ...]
  - artifacts/necessity_sufficiency_across.json
      { baseline: {type5_rate, not_type5_rate},
        addition: {alpha, not_type5_rate, delta_not_type5},
        ablation: {not_type5_rate, delta_not_type5},
        linear_probe_auc: {auc, ci95, site_key} }

CLI highlights:
  --direction_selection_decode sample|greedy             # validation decoding (default=sample)
  --val_temperature 0.8 --val_top_p 0.95 # sampling params
  --val_samples 5                        # draws per prompt for validation
  --constraint_aware                     # use constraint-aware judge rubric
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter

import numpy as np

from src.generate import greedy_with_intervention, sample_with_intervention, load_model
from src.label_constr_aware import label_batch
from src.utils import set_global_seed, softmax, kl_divergence, top_k_overlap, bootstrap_auc

# ------------------------ Candidate Vectors from Train Activations ------------------------

# OK!
def load_train_acts(acts_path: Path, meta_path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    acts = np.load(acts_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = {k: acts[k] for k in acts.files}
    return arrays, meta

# OK!
def build_candidates_type5(
    arrays: Dict[str, np.ndarray],
    meta: dict
) -> List[dict]:
    """
    For each site key (layer_{mid|late}_t{1..5}), compute per-prompt Δ_p:
        mean(h | Type-5) - mean(h | not Type-5),
    then average over prompts to get Δ, and build unit vector v plus magnitude ||Δ||.
    Returns list of candidates: {key, layer_name, layer_index, pos, v_unit, delta_mag}
    """
    samples = meta["samples"]
    idx_by_prompt = defaultdict(list)
    for row_idx, row in enumerate(samples):
        idx_by_prompt[row["prompt_id"]].append(row_idx)

    candidates = []
    for key, mat in arrays.items():
        m = re.match(r"layer_(mid|late)_t(\d+)$", key)
        if not m:
            continue
        layer_name = m.group(1)
        pos = int(m.group(2))

        deltas = []
        for pid, idxs in idx_by_prompt.items():
            types = [samples[i]["type"] for i in idxs]
            has_t5 = any(t == 5 for t in types)
            has_not = any((t != 5) for t in types)
            if not (has_t5 and has_not):
                continue
            rows = mat[idxs, :]  # [k, d_model]
            rows_t5 = rows[[i for i, t in enumerate(types) if t == 5], :]
            rows_not = rows[[i for i, t in enumerate(types) if t != 5], :]
            mu_t5 = rows_t5.mean(axis=0)
            mu_not = rows_not.mean(axis=0)
            deltas.append(mu_t5 - mu_not)
        if len(deltas) == 0:
            continue
        delta = np.mean(np.stack(deltas, axis=0), axis=0)
        mag = float(np.linalg.norm(delta) + 1e-12)
        v_unit = (delta / mag).astype(np.float32)
        candidates.append({
            "key": key,
            "layer_name": layer_name,
            "layer_index": meta["layer_indices"][layer_name],
            "pos": pos,
            "v_unit": v_unit,
            "delta_mag": mag
        })
    return candidates

# ------------------------ Prompt loading ------------------------

def load_all_prompts(prompt_dir: Path) -> Dict[str, dict]:
    out = {}
    for p in prompt_dir.glob("*.json"):
        try:
            arr = json.loads(p.read_text(encoding="utf-8"))
            for item in arr:
                out[item["id"]] = item
        except Exception:
            continue
    return out

def ids_from_split(path: Path) -> List[str]:
    arr = json.loads(path.read_text(encoding="utf-8"))
    ids = []
    for item in arr:
        if isinstance(item, dict) and "prompt_id" in item:
            ids.append(item["prompt_id"])
        elif isinstance(item, str):
            ids.append(item)
    return ids

# ------------------------ Evaluation helpers ------------------------

def eval_controls_drift(
    model, tokenizer, control_prompts: List[str],
    layer_index: int, v_unit: np.ndarray, alpha: float
) -> Tuple[float, float]:
    """Compute mean KL (first 5 tokens) and Top-10 overlap between baseline and steered (greedy)."""
    kls = []
    overlaps = []

    for text in control_prompts:
        # baseline logits (first 5)
        _, base_logits_list = greedy_with_intervention(
            model, tokenizer, text, layer_index=0, add_vec=None, ablate_vec=None,
            alpha=0.0, max_new_tokens=4
        )
        # steered logits (first 5)
        _, steer_logits_list = greedy_with_intervention(
            model, tokenizer, text, layer_index=layer_index, add_vec=v_unit, ablate_vec=None,
            alpha=alpha, max_new_tokens=4
        )
        L = min(len(base_logits_list), len(steer_logits_list), 5)
        for t in range(L):
            b = base_logits_list[t].squeeze()
            s = steer_logits_list[t].squeeze()
            p = softmax(b)
            q = softmax(s)
            kls.append(kl_divergence(p, q))
            overlaps.append(top_k_overlap(b, s, k=10))
    mean_kl = float(np.mean(kls)) if kls else 0.0
    mean_top10 = float(np.mean(overlaps)) if overlaps else 0.0
    return mean_kl, mean_top10

# Missing: produce per prompt responses labels distribution (it computes only aggregate rates now (prompts-wise))
def eval_validation_baseline(
    model, tokenizer, val_prompts: List[str],
    judge_model: str, constraint_aware: bool,
    decode_mode: str = "sample",
    temperature: float = 0.8,
    top_p: float = 0.95,
    num_samples: int = 5,
    base_seed: int = 1234,
) -> Tuple[float, float, int, List[int]]:
    """
    Compute baseline Type-5 rate on validation prompts.
    If decode_mode='sample', draw num_samples per prompt at (temperature, top_p).
    Returns (type5_rate, not_type5_rate, N_total_draws, labels_list).
    """
    pairs = []
    if decode_mode == "greedy":
        for text in val_prompts:
            out, _ = greedy_with_intervention(
                model, tokenizer, text, layer_index=0, add_vec=None, ablate_vec=None,
                alpha=0.0, max_new_tokens=200
            )
            pairs.append((text, out))
    else:
        for pi, text in enumerate(val_prompts): # 8 prompts
            for si in range(num_samples): # 10 samples per prompt --> 80 generations
                seed = base_seed + pi * 1000 + si
                out, _ = sample_with_intervention(
                    model, tokenizer, text, layer_index=0, add_vec=None, ablate_vec=None,
                    alpha=0.0, temperature=temperature, top_p=top_p, max_new_tokens=200, seed=seed
                )
                pairs.append((text, out))
                print(f"\n\n<<Prompt>> '{text}'\n<<Response>> '{out[:500]}'")
                print(f"{len(pairs)} responses sampled so far...")

    labels = label_batch(judge_model, pairs, constraint_aware=constraint_aware)

    N = len(labels)
    t5 = sum(1 for t in labels if t == 5)
    return float(t5 / max(1, N)), float(1.0 - t5 / max(1, N)), N, labels

# Missing: produce per prompt responses labels distribution (it computes only aggregate rates now (prompts-wise))
def eval_validation_rates(
    model, tokenizer, val_prompts: List[str],
    layer_index: int, v_unit: np.ndarray, alpha: float,
    judge_model: str, constraint_aware: bool,
    decode_mode: str = "sample",
    temperature: float = 0.8,
    top_p: float = 0.95,
    num_samples: int = 5,
    base_seed: int = 5678,
) -> Tuple[float, float, int]:
    """
    Return (type5_rate, not_type5_rate, N_total_draws) for steered validation.
    If decode_mode='sample', draw num_samples per prompt with fixed seeds for reproducibility.
    """
    pairs = []
    if decode_mode == "greedy":
        for text in val_prompts:
            out, _ = greedy_with_intervention(
                model, tokenizer, text, layer_index, add_vec=v_unit, ablate_vec=None,
                alpha=alpha, max_new_tokens=200
            )
            pairs.append((text, out))
    else:
        for pi, text in enumerate(val_prompts):
            for si in range(num_samples):
                seed = base_seed + pi * 1000 + si
                out, _ = sample_with_intervention(
                    model, tokenizer, text, layer_index, add_vec=v_unit, ablate_vec=None, alpha=alpha,
                    temperature=temperature, top_p=top_p, max_new_tokens=200, seed=seed
                )
                pairs.append((text, out))
                print(f"\n\n<<Prompt>> '{text}'\n<<Response>> '{out[:500]}'")
                print(f"{len(pairs)} responses sampled so far...")

    labels = label_batch(judge_model, pairs, constraint_aware=constraint_aware)
    N = len(labels)
    t5 = sum(1 for t in labels if t == 5)
    rate5 = t5 / max(1, N)
    not5 = 1.0 - rate5
    return float(rate5), float(not5), N

# ------------------------ Main pipeline ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./models/llama-3.1-8b-instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--acts_path", default="data/acts/across/train.npz")
    ap.add_argument("--meta_path", default="data/acts/across/train_meta.json")
    ap.add_argument("--prompt_dir", default="prompts")
    ap.add_argument("--val_split", default="data/splits/conflict_validation.json")
    ap.add_argument("--controls_split", default="data/splits/controls_gold.json")
    ap.add_argument("--judge_model", default="deepseek/deepseek-chat-v3.1:free")
    ap.add_argument("--constraint_aware", action="store_true")
    ap.add_argument("--alphas", default="0.2,0.4")

    # NEW: validation decoding config
    ap.add_argument("--direction_selection_decode", default="sample", choices=["sample", "greedy"],
                    help="Decoding mode for direction selection via difference-in-means (default: sample).")
    ap.add_argument("--val_temperature", type=float, default=0.8)
    ap.add_argument("--val_top_p", type=float, default=0.95)
    ap.add_argument("--val_samples", type=int, default=10,
                    help="Number of draws per validation prompt when using sampling.")

    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    set_global_seed(args.seed)

    # Load model/tokenizer
    model, tokenizer = load_model(args.model_path, args.device)

    # Load train activations
    arrays, meta = load_train_acts(Path(args.acts_path), Path(args.meta_path))
    timesteps = meta.get("timesteps", [1, 2, 3, 4, 5])
    layer_indices = meta["layer_indices"]
    print(f"Loaded activations keys: {list(arrays.keys())}..., n={meta.get('n')}, d_model={meta.get('d_model')}")
    print(f"Layer indices: {layer_indices}; timesteps: {timesteps}")

    # Build candidates
    candidates = build_candidates_type5(arrays, meta)
    if not candidates:
        raise RuntimeError("No valid candidates could be built (need prompts with both Type-5 and non-Type-5 in train).")
    print(f"Built {len(candidates)} candidate vectors.")

    # Load prompts and splits
    all_prompts = load_all_prompts(Path(args.prompt_dir))
    val_ids = ids_from_split(Path(args.val_split))
    ctrl_ids = ids_from_split(Path(args.controls_split))

    val_prompts = []
    for pid in val_ids:
        rec = all_prompts.get(pid)
        if not rec:
            raise ValueError(f"Validation prompt_id not found in prompts: {pid}")
        val_prompts.append(rec["text"])

    ctrl_prompts = []
    for pid in ctrl_ids:
        rec = all_prompts.get(pid)
        if not rec:
            raise ValueError(f"Control prompt_id not found in prompts: {pid}")
        ctrl_prompts.append(rec["text"])

    # WE DON'T NEED VALIDATION BASELINES WE HAVE ALREADY DEV and VALIDATION RESPONSES GENERATED and LABELED
    # Baseline validation rates (sampling or greedy per flags)
    base_t5, base_not5, N_val, base_labels = eval_validation_baseline(
        model, tokenizer, val_prompts, args.judge_model, args.constraint_aware,
        decode_mode=args.direction_selection_decode, temperature=args.val_temperature, top_p=args.val_top_p,
        num_samples=args.val_samples, base_seed=1234
    )
    print(f"Validation baseline ({args.direction_selection_decode}): Type-5 rate={base_t5:.3f}, ¬Type-5 rate={base_not5:.3f}, N={N_val}")

    # Evaluate candidates across alphas
    alphas = [float(x) for x in args.alphas.split(",")] # 2 alphas
    sel_rows = []

    for cand in candidates: # 10 candidates
        v = cand["v_unit"]
        lidx = cand["layer_index"]
        lname = cand["layer_name"]
        pos = cand["pos"]

        for a in alphas: # 20 loops * 20 generations = 400 (gen + label)
            # Validation (steered) — sampling/greedy per flags
            rate5_s, not5_s, N_s = eval_validation_rates(
                model, tokenizer, val_prompts, lidx, v, a, args.judge_model, args.constraint_aware,
                decode_mode=args.direction_selection_decode, temperature=args.val_temperature, top_p=args.val_top_p,
                num_samples=args.val_samples, base_seed=5678
            )
            d_type5 = rate5_s - base_t5
            d_not5 = not5_s - base_not5

            # Controls drift (greedy, stable)
            mean_kl, mean_top10 = eval_controls_drift(model, tokenizer, ctrl_prompts, lidx, v, a)

            J = (-d_not5) - 1.0 * mean_kl  # per plan

            row = {
                "layer_name": lname,
                "layer_index": lidx,
                "pos": pos,
                "alpha": a,
                "delta_Type5": d_type5,
                "delta_not_Type5": d_not5,
                "mean_KL_controls": mean_kl,
                "Top10_overlap_controls": mean_top10,
                "J_multi": J,
                "n_val": N_s,
                "n_ctrl": len(ctrl_prompts),
                "decode mode": args.direction_selection_decode
            }
            sel_rows.append(row)
            print(f"[{lname}@t{pos} | α={a:.2f}] ΔT5={d_type5:+.3f} Δ¬T5={d_not5:+.3f} KL={mean_kl:.3f} Top10={mean_top10:.3f} J={J:+.3f}")
    
    # Select best at α=0.2
    rows_a02 = [r for r in sel_rows if abs(r["alpha"] - 0.2) < 1e-6]
    # Select best at α=0.4
    rows_a04 = [r for r in sel_rows if abs(r["alpha"] - 0.4) < 1e-6]

    if not rows_a04:
        raise RuntimeError("No rows found for α=0.4")
    if not rows_a02:
        raise RuntimeError("No rows found for α=0.2")
    best_a04 = max(rows_a04, key=lambda r: r["J_multi"])
    print("\nWinner @ α=0.4:")
    print(json.dumps(best_a04, indent=2))

    best_a02 = max(rows_a02, key=lambda r: r["J_multi"])
    print("\nWinner @ α=0.2:")
    print(json.dumps(best_a02, indent=2))

    # Retrieve the winning candidate vector
    win_layer_a04 = best_a04["layer_name"]
    win_pos_a04 = best_a04["pos"]
    win_key_a04 = f"layer_{win_layer_a04}_t{win_pos_a04}"
    win_cand_a04 = None
    for c in candidates:
        if c["key"] == win_key_a04:
            win_cand_a04 = c
            break
    assert win_cand_a04 is not None

    win_layer_a02 = best_a02["layer_name"]
    win_pos_a02 = best_a02["pos"]
    win_key_a02 = f"layer_{win_layer_a02}_t{win_pos_a02}"
    win_cand_a02 = None
    for c in candidates:
        if c["key"] == win_key_a02:
            win_cand_a02 = c
            break
    assert win_cand_a02 is not None

    # Choose between α=0.2 and α=0.4 based on J_multi
    if best_a04 > best_a02:
        win_layer = win_layer_a04
        win_key = win_key_a04
        win_cand = win_cand_a04
        alpha_star = 0.4
        v_star = win_cand_a04["v_unit"]
        l_star = best_a04["layer_index"]
        pos_star = win_pos_a04
    else:
        win_layer = win_layer_a02
        win_key = win_key_a02
        win_cand = win_cand_a02
        alpha_star = 0.2
        v_star = win_cand_a02["v_unit"]
        l_star = best_a02["layer_index"]
        pos_star = win_pos_a02

    # Necessity/sufficiency sanity (greedy decoding for determinism):
    # Addition should decrease ¬Type-5; ablation should increase ¬Type-5.
    # Baseline (greedy)
    pairs_base = []
    for text in val_prompts:
        out, _ = greedy_with_intervention(
            model, tokenizer, text, layer_index=0, add_vec=None, ablate_vec=None,
            alpha=0.0, max_new_tokens=200
        )
        pairs_base.append((text, out))
    base_labels = label_batch(args.judge_model, pairs_base, constraint_aware=args.constraint_aware)
    base_t5 = sum(1 for t in base_labels if t == 5) / max(1, len(base_labels))
    base_not5 = 1.0 - base_t5
    print(f"\nGreedy baseline: Type-5 rate={base_t5:.3f}, ¬Type-5 rate={base_not5:.3f}, N={len(base_labels)}")

    # Addition (greedy)
    pairs_add = []
    for text in val_prompts:
        out, _ = greedy_with_intervention(
            model, tokenizer, text, l_star, add_vec=v_star, ablate_vec=None, alpha=alpha_star, max_new_tokens=200
        )
        pairs_add.append((text, out))
    add_labels = label_batch(args.judge_model, pairs_add, constraint_aware=args.constraint_aware)
    not5_add = 1.0 - (sum(1 for t in add_labels if t == 5) / max(1, len(add_labels)))
    delta_not5_add = not5_add - base_not5  # expect negative

    # Ablation (greedy)
    pairs_abl = []
    for text in val_prompts:
        out, _ = greedy_with_intervention(
            model, tokenizer, text, l_star, add_vec=None, ablate_vec=v_star, alpha=0.0, max_new_tokens=200
        )
        pairs_abl.append((text, out))
    abl_labels = label_batch(args.judge_model, pairs_abl, constraint_aware=args.constraint_aware)
    not5_abl = 1.0 - (sum(1 for t in abl_labels if t == 5) / max(1, len(abl_labels)))
    delta_not5_abl = not5_abl - base_not5  # expect positive

    # Linear-probe AUC at chosen site using v* ⋅ h as score (train set)
    feats = arrays[win_key]  # [n, d]
    y = np.array([1 if s["type"] == 5 else 0 for s in meta["samples"]], dtype=np.int32)
    scores = feats @ (v_star / (np.linalg.norm(v_star) + 1e-12))
    scores_pos = scores[y == 1]
    scores_neg = scores[y == 0]
    if len(scores_pos) > 0 and len(scores_neg) > 0:
        auc, (lo, hi) = bootstrap_auc(scores_pos, scores_neg, B=500, seed=args.seed)
    else:
        raise RuntimeError("Positive or Negative samples not available, hence AUC is not computable.")

    # Save artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "v_star_across.npz",
        vector=v_star.astype(np.float32),
        layer_name=win_layer,
        layer_index=np.int32(l_star),
        pos=np.int32(pos_star),
        alpha=np.float32(alpha_star),
        delta_magnitude=np.float32(win_cand["delta_mag"]),
        timesteps=np.array(timesteps, dtype=np.int32)
    )

    (out_dir / "selection_table_across.json").write_text(json.dumps(sel_rows, indent=2), encoding="utf-8")

    nec_suf = {
        "baseline": {"type5_rate": float(base_t5), "not_type5_rate": float(base_not5),
                     "decode mode": "greedy"},
        "addition": {"alpha": float(alpha_star), "not_type5_rate": float(not5_add), "delta_not_type5": float(delta_not5_add), "decode mode": "greedy"},
        "ablation": {"not_type5_rate": float(not5_abl), "delta_not_type5": float(delta_not5_abl), "decode mode": "greedy"},
        "linear_probe_auc": {"auc": float(auc), "ci95": [float(lo), float(hi)], "site_key": win_key}
    }
    (out_dir / "necessity_sufficiency_across.json").write_text(json.dumps(nec_suf, indent=2), encoding="utf-8")

    # Final prints (acceptance-like)
    print("\n=== ACCEPTANCE SUMMARY ===")
    print(f"AUC (Type-5 vs ¬Type-5) at {win_key}: {auc:.3f} (95% CI {lo:.3f}–{hi:.3f})")
    print(f"Addition Δ¬Type-5 (want < 0): {delta_not5_add:+.3f}")
    print(f"Ablation Δ¬Type-5 (want > 0): {delta_not5_abl:+.3f}")
    print(f"Artifacts written to: {out_dir}")
    print("\nREPRO CMD:")
    print(f"python src/vector_multi_across.py "
          f"--model_path {args.model_path} --device {args.device} "
          f"--acts_path {args.acts_path} --meta_path {args.meta_path} "
          f"--prompt_dir {args.prompt_dir} --val_split {args.val_split} --controls_split {args.controls_split} "
          f"--judge_model {args.judge_model} {'--constraint_aware' if args.constraint_aware else ''} "
          f"--direction_selection_decode {args.direction_selection_decode} --val_temperature {args.val_temperature} --val_top_p {args.val_top_p} --val_samples {args.val_samples} "
          f"--alphas {args.alphas} --out_dir {args.out_dir}"
    )

if __name__ == "__main__":
    main()
