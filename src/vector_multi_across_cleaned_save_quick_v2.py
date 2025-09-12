#!/usr/bin/env python3
"""
Step 5-vector-multi-across (Updated for stochastic validation and fast-test path)

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
  --direction_selection_decode sample|greedy
  --val_temperature 0.8 --val_top_p 0.95
  --val_samples 5
  --constraint_aware
  --alphas "0.15,0.3,0.45"   # any set; winner is picked from these
  --quick                    # optional: tiny run (2 prompts, 2 draws) to sanity check E2E
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np

from src.generate import greedy_with_intervention, sample_with_intervention, load_model
from src.label_constr_aware import (
    label_batch, label_batch_full, save_labeled_results,
    labels_exist, load_final_labels, type5_rate_from_finals
)
from src.utils import set_global_seed, softmax, kl_divergence, top_k_overlap, bootstrap_auc

# ------------------------ IO helpers for cached generations ------------------------

def _gens_path(base_dir: Path, tag: str) -> Path:
    return base_dir / f"{tag}_gens.jsonl"

def _save_gens(base_dir: Path, tag: str, samples: List[dict]) -> None:
    """Write generated samples (prompt_id, sample_idx, text, output_text) to JSONL."""
    base_dir.mkdir(parents=True, exist_ok=True)
    p = _gens_path(base_dir, tag)
    with p.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"💾 Saved generations to: {p}")

def _load_gens(base_dir: Path, tag: str) -> List[dict]:
    """Load previously generated samples from JSONL, preserving fields."""
    p = _gens_path(base_dir, tag)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ------------------------ Candidate Vectors from Train Activations ------------------------

def load_train_acts(acts_path: Path, meta_path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    acts = np.load(acts_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = {k: acts[k] for k in acts.files}
    return arrays, meta

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

def eval_validation_baseline(
    model, tokenizer, val_prompts: list[str],
    judge_model: str, constraint_aware: bool,
    decode_mode: str = "sample",
    temperature: float = 0.8,
    top_p: float = 0.95,
    num_samples: int = 5,
    base_seed: int = 1234,
    val_ids: list[str] | None = None,
    return_payload: bool = False,
    cache_dir: str | None = None,
    cache_tag: str | None = None,
):
    """
    Baseline (no intervention) rates on the validation prompts.

    Caching order:
      1) If labels exist: load and return metrics from them (no gens/labeling).
      2) Else if gens exist: load gens and run labeling only.
      3) Else: generate -> save gens -> label -> save labels.
    """
    base_dir = Path(cache_dir) if cache_dir else None

    # (1) labels already cached?
    if base_dir and cache_tag and labels_exist(base_dir, cache_tag):
        finals = load_final_labels(base_dir, cache_tag)
        rate5, not5, N = type5_rate_from_finals(finals)
        print(f"🔁 Baseline cached labels: {cache_tag}  N={N}  Type-5={rate5:.3f}")
        labels_int = [r.get("type", 7) for r in finals]
        return rate5, not5, N, labels_int, None, None

    samples: List[dict] = []
    pairs: List[Tuple[str, str]] = []

    # (2) generations cached?
    gens_loaded = False
    if base_dir and cache_tag:
        gens = _load_gens(base_dir, cache_tag)
        if gens:
            print(f"🔁 Baseline cached generations found: {_gens_path(base_dir, cache_tag)}  ({len(gens)} rows)")
            samples = gens
            pairs = [(s["text"], s["output_text"]) for s in samples]
            gens_loaded = True

    # (3) else generate now and save gens
    if not gens_loaded:
        if decode_mode == "greedy":
            for pi, text in enumerate(val_prompts):
                out, _ = greedy_with_intervention(
                    model, tokenizer, text, layer_index=0, add_vec=None, ablate_vec=None,
                    alpha=0.0, max_new_tokens=200
                )
                pid = val_ids[pi] if val_ids else f"VAL_{pi}"
                samples.append({"prompt_id": pid, "sample_idx": 0, "text": text, "output_text": out})
                pairs.append((text, out))
                print(f"\nBaseline gen (greedy): prompt {pi+1}/{len(val_prompts)}\n{text}\n==>\n{out}\n")
        else:
            for pi, text in enumerate(val_prompts):
                for si in range(num_samples):
                    seed = base_seed + pi * 1000 + si
                    out, _ = sample_with_intervention(
                        model, tokenizer, text, layer_index=0, add_vec=None, ablate_vec=None,
                        alpha=0.0, temperature=temperature, top_p=top_p, max_new_tokens=200, seed=seed
                    )
                    pid = val_ids[pi] if val_ids else f"VAL_{pi}"
                    samples.append({"prompt_id": pid, "sample_idx": si, "text": text, "output_text": out})
                    pairs.append((text, out))
                    print(f"\nBaseline gen (sample): prompt {pi+1}/{len(val_prompts)} sample {si+1}/{num_samples}\n{text}\n==>\n{out}\n")
        if base_dir and cache_tag:
            _save_gens(base_dir, cache_tag, samples)

    # ---- Label ints for metrics (with retry) ----
    try:
        labels = label_batch(judge_model, pairs, constraint_aware=constraint_aware)
    except Exception as e:
        import time
        print(f"Labeling failed with {e}, short retrying --- up to 5 times after 10s...")
        labels = None
        for attempt in range(5):
            time.sleep(10)
            try:
                labels = label_batch(judge_model, pairs, constraint_aware=constraint_aware)
                print("Retry succeeded.")
                break
            except Exception as e2:
                print(f"Short retry {attempt+1} failed with {e2}.")
        if labels is None:
            print(f"Labeling failed with {e}, long retrying --- up to 5 times after 120s...")
            for attempt in range(5):
                time.sleep(120)
                try:
                    labels = label_batch(judge_model, pairs, constraint_aware=constraint_aware)
                    print("Retry succeeded.")
                    break
                except Exception as e2:
                    print(f"Long retry {attempt+1} failed with {e2}.")
        if labels is None:
            raise RuntimeError("Labeling failed after short and long retries.")

    N = len(labels)
    t5 = sum(1 for t in labels if t == 5)
    rate5 = float(t5 / max(1, N))
    not5 = float(1.0 - rate5)

    # Save labeled results
    if base_dir and cache_tag:
        labeled_full = label_batch_full(judge_model, samples, constraint_aware=constraint_aware)
        save_labeled_results(base_dir, cache_tag, labeled_full, constraint_aware)

    if return_payload:
        labeled_full = label_batch_full(judge_model, samples, constraint_aware=constraint_aware)
        return rate5, not5, N, labels, samples, labeled_full

    return rate5, not5, N, labels, None, None

def eval_validation_rates(
    model, tokenizer, val_prompts: list[str], val_ids: list[str],
    layer_index: int, v_unit: np.ndarray, alpha: float,
    judge_model: str, constraint_aware: bool,
    decode_mode: str = "sample",
    temperature: float = 0.8,
    top_p: float = 0.95,
    num_samples: int = 5,
    base_seed: int = 5678,
    save_dir: str | None = None,
    run_tag: str | None = None,
):
    """
    Steered validation:

    Caching order:
      1) If labels exist: load and return metrics from them (no gens/labeling).
      2) Else if gens exist: load gens and run labeling only.
      3) Else: generate -> save gens -> label -> save labels.
    """
    base_dir = Path(save_dir) if save_dir else None

    # (1) labels already cached?
    if base_dir and run_tag and labels_exist(base_dir, run_tag):
        finals = load_final_labels(base_dir, run_tag)
        rate5, not5, N = type5_rate_from_finals(finals)
        print(f"🔁 Steered cached labels: {run_tag}  N={N}  Type-5={rate5:.3f}")
        return rate5, not5, N

    samples: List[dict] = []
    pairs: List[Tuple[str, str]] = []

    # (2) generations cached?
    gens_loaded = False
    if base_dir and run_tag:
        gens = _load_gens(base_dir, run_tag)
        if gens:
            print(f"🔁 Steered cached generations found: {_gens_path(base_dir, run_tag)}  ({len(gens)} rows)")
            samples = gens
            pairs = [(s["text"], s["output_text"]) for s in samples]
            gens_loaded = True

    # (3) else generate now and save gens
    if not gens_loaded:
        if decode_mode == "greedy":
            for pi, text in enumerate(val_prompts):
                out, _ = greedy_with_intervention(
                    model, tokenizer, text, layer_index, add_vec=v_unit, ablate_vec=None,
                    alpha=alpha, max_new_tokens=200
                )
                pid = val_ids[pi] if val_ids else f"VAL_{pi}"
                samples.append({"prompt_id": pid, "sample_idx": 0, "text": text, "output_text": out})
                pairs.append((text, out))
                print(f"\nSteered gen (greedy): prompt {pi+1}/{len(val_prompts)}\n{text}\n==>\n{out}\n")
        else:
            for pi, text in enumerate(val_prompts):
                for si in range(num_samples):
                    seed = base_seed + pi * 1000 + si
                    out, _ = sample_with_intervention(
                        model, tokenizer, text, layer_index, add_vec=v_unit, ablate_vec=None, alpha=alpha,
                        temperature=temperature, top_p=top_p, max_new_tokens=200, seed=seed
                    )
                    pid = val_ids[pi] if val_ids else f"VAL_{pi}"
                    samples.append({"prompt_id": pid, "sample_idx": si, "text": text, "output_text": out})
                    pairs.append((text, out))
                    print(f"\nSteered gen (sample): prompt {pi+1}/{len(val_prompts)} sample {si+1}/{num_samples}\n{text}\n==>\n{out}\n")
        if base_dir and run_tag:
            _save_gens(base_dir, run_tag, samples)

    # ---- Label ints for metrics (with retry) ----
    try:
        labels = label_batch(judge_model, pairs, constraint_aware=constraint_aware)
    except Exception as e:
        import time
        print(f"Labeling failed with {e}, short retrying --- up to 5 times after 10s...")
        labels = None
        for attempt in range(5):
            time.sleep(10)
            try:
                labels = label_batch(judge_model, pairs, constraint_aware=constraint_aware)
                print("Retry succeeded.")
                break
            except Exception as e2:
                print(f"Short retry {attempt+1} failed with {e2}.")
        if labels is None:
            print(f"Labeling failed with {e}, long retrying --- up to 5 times after 120s...")
            for attempt in range(5):
                time.sleep(120)
                try:
                    labels = label_batch(judge_model, pairs, constraint_aware=constraint_aware)
                    print("Retry succeeded.")
                    break
                except Exception as e2:
                    print(f"Long retry {attempt+1} failed with {e2}.")
        if labels is None:
            raise RuntimeError("Labeling failed after short and long retries.")

    N = len(labels)
    t5 = sum(1 for t in labels if t == 5)
    rate5 = t5 / max(1, N)
    not5 = 1.0 - rate5

    # Save labeled results
    if base_dir and run_tag:
        labeled_full = label_batch_full(judge_model, samples, constraint_aware=constraint_aware)
        save_labeled_results(base_dir, run_tag, labeled_full, constraint_aware)

    return float(rate5), float(not5), N

# ------------------------ Main pipeline ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./models/llama-3.1-8b-instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--acts_path", default="data/acts/across/train.npz")
    ap.add_argument("--meta_path", default="data/acts/across/train_meta.json")
    ap.add_argument("--prompt_dir", default="prompts/separable_constraints")
    ap.add_argument("--val_split", default="data/splits_across/conflict_validation.json")
    ap.add_argument("--controls_split", default="data/splits_across/controls_gold.json")
    ap.add_argument("--judge_model", default="deepseek/deepseek-chat-v3.1:free")
    ap.add_argument("--constraint_aware", action="store_true")
    ap.add_argument("--alphas", default="0.2,0.4")
    ap.add_argument("--intervention_dir", default="data/interventions")
    ap.add_argument("--quick", action="store_true", help="Fast test: keep only 2 prompts & 2 draws; use only mid_t1,t2 acts.")

    # Validation decoding config
    ap.add_argument("--direction_selection_decode", default="sample", choices=["sample", "greedy"])
    ap.add_argument("--val_temperature", type=float, default=0.8)
    ap.add_argument("--val_top_p", type=float, default=0.95)
    ap.add_argument("--val_samples", type=int, default=10)

    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    INTERV_DIR = Path(args.intervention_dir)
    INTERV_DIR.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)

    # Load model/tokenizer
    model, tokenizer = load_model(args.model_path, args.device)

    # Load train activations
    arrays, meta = load_train_acts(Path(args.acts_path), Path(args.meta_path))

    kept_ts = None
    if args.quick:
        # === [FAST-TEST] Keep only layer_mid_t1 and layer_mid_t2 ===
        keep_keys = {"layer_mid_t1", "layer_mid_t2"}
        arrays = {k: v for k, v in arrays.items() if k in keep_keys}
        kept_ts = sorted({int(re.match(r"layer_(mid|late)_t(\d+)$", k).group(2))
                          for k in arrays.keys() if re.match(r"layer_(mid|late)_t(\d+)$", k)})
    timesteps = kept_ts if kept_ts else meta.get("timesteps", [1, 2, 3, 4, 5])

    layer_indices = meta["layer_indices"]
    print(f"Loaded activations keys (kept): {list(arrays.keys())}, n={meta.get('n')}, d_model={meta.get('d_model')}")
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

    # === [FAST-TEST] Shrink sets & draws if --quick ===
    if args.quick:
        val_prompts = val_prompts[:2]
        val_ids = val_ids[:2]
        ctrl_prompts = ctrl_prompts[:2]
        args.val_samples = min(args.val_samples, 2)
        print("⚡ QUICK MODE: using 2 validation prompts, 2 control prompts, 2 samples per prompt.")

    # 1) Baseline with cache (now gens-aware)
    base_tag = f"baseline_val_{args.direction_selection_decode}_T{args.val_temperature}_P{args.val_top_p}_S{args.val_samples}"
    base_t5, base_not5, N_val, base_labels, _, _ = eval_validation_baseline(
        model, tokenizer, val_prompts, args.judge_model, args.constraint_aware,
        decode_mode=args.direction_selection_decode, temperature=args.val_temperature,
        top_p=args.val_top_p, num_samples=args.val_samples, base_seed=1234,
        val_ids=val_ids, return_payload=False,
        cache_dir=str(INTERV_DIR), cache_tag=base_tag
    )
    print(f"Validation baseline: Type-5={base_t5:.3f}, ¬Type-5={base_not5:.3f}, N={N_val}")

    # 2) Candidate × alpha selection — flexible alphas
    alphas = [float(x) for x in args.alphas.split(",") if x.strip() != ""]
    sel_rows = []

    for cand in candidates:
        v = cand["v_unit"]; lidx = cand["layer_index"]; lname = cand["layer_name"]; pos = cand["pos"]

        for a in alphas:
            run_tag = f"val_l{lidx}_{lname}_t{pos}_a{a:.2f}_{args.direction_selection_decode}"

            # Load-or-run steered validation (now gens-aware)
            rate5_s, not5_s, N_s = eval_validation_rates(
                model, tokenizer, val_prompts, val_ids,
                lidx, v, a, args.judge_model, args.constraint_aware,
                decode_mode=args.direction_selection_decode,
                temperature=args.val_temperature, top_p=args.val_top_p,
                num_samples=args.val_samples, base_seed=5678,
                save_dir=str(INTERV_DIR), run_tag=run_tag
            )

            d_type5 = rate5_s - base_t5
            d_not5  = not5_s - base_not5

            # Controls drift: cache per (lidx, pos, alpha)
            drift_tag = f"ctrl_l{lidx}_{lname}_t{pos}_a{a:.2f}"
            drift_file = INTERV_DIR / f"{drift_tag}_controls.json"
            if drift_file.exists():
                drift = json.loads(drift_file.read_text(encoding="utf-8"))
                mean_kl, mean_top10 = drift["mean_kl"], drift["mean_top10"]
                print(f"🔁 Controls cached: {drift_tag} KL={mean_kl:.3f} Top10={mean_top10:.3f}")
            else:
                mean_kl, mean_top10 = eval_controls_drift(model, tokenizer, ctrl_prompts, lidx, v, a)
                drift_file.write_text(
                    json.dumps({"mean_kl": float(mean_kl), "mean_top10": float(mean_top10)}, indent=2),
                    encoding="utf-8"
                )

            J = (-d_not5) - 1.0 * mean_kl
            sel_rows.append({
                "layer_name": lname, "layer_index": lidx, "pos": pos, "alpha": a,
                "delta_Type5": d_type5, "delta_not_Type5": d_not5,
                "mean_KL_controls": mean_kl, "Top10_overlap_controls": mean_top10,
                "J_multi": J, "n_val": N_s, "n_ctrl": len(ctrl_prompts),
                "decode mode": args.direction_selection_decode,
                "saved_to": f"{INTERV_DIR / (run_tag + '_labels.jsonl')}"
            })
            print(f"[{lname}@t{pos} | α={a:.2f}] ΔT5={d_type5:+.3f} Δ¬T5={d_not5:+.3f} KL={mean_kl:.3f} Top10={mean_top10:.3f} J={J:+.3f}")

    # ---- FLEXIBLE SELECTION: pick the single best (candidate, alpha) across the provided alphas ----
    if not sel_rows:
        raise RuntimeError("No selection rows computed.")
    best_any = max(sel_rows, key=lambda r: r["J_multi"])
    print("\nWinner (best J_multi across provided alphas):")
    print(json.dumps(best_any, indent=2))

    win_layer = best_any["layer_name"]
    l_star    = best_any["layer_index"]
    pos_star  = best_any["pos"]
    alpha_star = float(best_any["alpha"])
    win_key   = f"layer_{win_layer}_t{pos_star}"

    # Retrieve winning vector
    win_cand = None
    for c in candidates:
        if c["key"] == win_key:
            win_cand = c
            break
    assert win_cand is not None
    v_star = win_cand["v_unit"]

    # Necessity/sufficiency sanity (greedy decoding for determinism):
    # Baseline (greedy)
    base_tag = f"greedy_base"
    if not labels_exist(INTERV_DIR, base_tag):
        samples_base = []
        for pi, text in enumerate(val_prompts):
            out, _ = greedy_with_intervention(
                model, tokenizer, text, layer_index=0, add_vec=None, ablate_vec=None,
                alpha=0.0, max_new_tokens=200
            )
            samples_base.append({"prompt_id": val_ids[pi], "sample_idx": 0, "text": text, "output_text": out})
        base_labels_full = label_batch_full(args.judge_model, samples_base, constraint_aware=args.constraint_aware)
        save_labeled_results(INTERV_DIR, base_tag, base_labels_full, args.constraint_aware)
    else:
        print(f"🔁 Skipping greedy baseline: cached {base_tag}")
        base_labels_full = load_final_labels(INTERV_DIR, base_tag)

    base_t5 = sum(1 for t in base_labels_full if t["type"] == 5) / max(1, len(base_labels_full))
    base_not5 = 1.0 - base_t5
    print(f"\nGreedy baseline: Type-5 rate={base_t5:.3f}, ¬Type-5 rate={base_not5:.3f}, N={len(base_labels_full)}")

    # Addition (greedy)
    add_tag = f"greedy_add_l{l_star}_{win_layer}_t{pos_star}_a{alpha_star:.2f}"
    if not labels_exist(INTERV_DIR, add_tag):
        samples_add = []
        for pi, text in enumerate(val_prompts):
            out, _ = greedy_with_intervention(model, tokenizer, text, l_star, add_vec=v_star, ablate_vec=None, alpha=alpha_star, max_new_tokens=200)
            samples_add.append({"prompt_id": val_ids[pi], "sample_idx": 0, "text": text, "output_text": out})
        add_labels_full = label_batch_full(args.judge_model, samples_add, constraint_aware=args.constraint_aware)
        save_labeled_results(INTERV_DIR, add_tag, add_labels_full, args.constraint_aware)
    else:
        print(f"🔁 Skipping greedy addition: cached {add_tag}")
        add_labels_full = load_final_labels(INTERV_DIR, add_tag)

    not5_add = 1.0 - (sum(1 for t in add_labels_full if t["type"] == 5) / max(1, len(add_labels_full)))
    delta_not5_add = not5_add - base_not5  # expect negative

    # Ablation (greedy)
    abl_tag = f"greedy_ablate_l{l_star}_{win_layer}_t{pos_star}"
    if not labels_exist(INTERV_DIR, abl_tag):
        samples_abl = []
        for pi, text in enumerate(val_prompts):
            out, _ = greedy_with_intervention(
                model, tokenizer, text, l_star, add_vec=None, ablate_vec=v_star, alpha=0.0, max_new_tokens=200
            )
            samples_abl.append({"prompt_id": val_ids[pi], "sample_idx": 0, "text": text, "output_text": out})
        abl_labels_full = label_batch_full(args.judge_model, samples_abl, constraint_aware=args.constraint_aware)
        save_labeled_results(INTERV_DIR, abl_tag, abl_labels_full, args.constraint_aware)
    else:
        print(f"🔁 Skipping greedy ablation: cached {abl_tag}")
        abl_labels_full = load_final_labels(INTERV_DIR, abl_tag)

    not5_abl = 1.0 - (sum(1 for t in abl_labels_full if t["type"] == 5) / max(1, len(abl_labels_full)))
    delta_not5_abl = not5_abl - base_not5  # expect positive

    # Linear-probe AUC at chosen site using v* ⋅ h as score (train set)
    feats = arrays[win_key]  # [n, d]
    y = np.array([1 if s["type"] == 5 else 0 for s in meta["samples"]], dtype=np.int32)
    scores = feats @ (v_star / (np.linalg.norm(v_star) + 1e-12))
    scores_pos = scores[y == 1]
    scores_neg = scores[y == 0]
    if len(scores_pos) > 0 and len(scores_neg) > 0:
        B = 100 if args.quick else 500
        auc, (lo, hi) = bootstrap_auc(scores_pos, scores_neg, B=B, seed=args.seed)
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

    # Final prints
    print("\n=== ACCEPTANCE SUMMARY ===")
    print(f"AUC (Type-5 vs ¬Type-5) at {win_key}: {auc:.3f} (95% CI {lo:.3f}–{hi:.3f})")
    print(f"Winner: {win_layer}@t{pos_star} α={alpha_star:.2f}")
    print(f"Addition Δ¬Type-5 (want < 0): {delta_not5_add:+.3f}")
    print(f"Ablation Δ¬Type-5 (want > 0): {delta_not5_abl:+.3f}")
    print(f"Artifacts written to: {out_dir}")
    print("\nREPRO CMD:")
    print(f"python src/vector_multi_across.py "
          f"--model_path {args.model_path} --device {args.device} "
          f"--acts_path {args.acts_path} --meta_path {args.meta_path} "
          f"--prompt_dir {args.prompt_dir} --val_split {args.val_split} --controls_split {args.controls_split} "
          f"--judge_model {args.judge_model} {'--constraint_aware' if args.constraint_aware else ''} "
          f"--direction_selection_decode {args.direction_selection_decode} "
          f"--val_temperature {args.val_temperature} --val_top_p {args.val_top_p} --val_samples {args.val_samples} "
          f"--alphas {args.alphas} --out_dir {args.out_dir} {'--quick' if args.quick else ''}"
    )

if __name__ == "__main__":
    main()