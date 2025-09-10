#!/usr/bin/env python3
"""
Step 5-vector-multi-across (Updated for stochastic validation and bfloat16 fix)

Build multi-answers (Type-5) direction v* via candidate grid + validation selection.

Inputs:
  - data/acts/across/train.npz
      keys: layer_mid_t1..layer_late_t3
      values: [n, d_model]
  - data/acts/across/train_meta.json
      {
        "layer_indices": {"mid": int, "late": int},
        "timesteps": [1,2,3],
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
  --val_decode sample|greedy             # validation decoding (default=sample)
  --val_temperature 0.8 --val_top_p 0.95 # sampling params
  --val_samples 5                        # draws per prompt for validation
  --constraint_aware                     # use constraint-aware judge rubric
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter

import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# ------------------------ Utils ------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def kl_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())

def topk_overlap_idx(p_logits: np.ndarray, q_logits: np.ndarray, k: int = 10) -> float:
    p_top = set(np.argsort(p_logits)[-k:])
    q_top = set(np.argsort(q_logits)[-k:])
    return len(p_top & q_top) / float(k)

def bootstrap_auc(scores_pos: np.ndarray, scores_neg: np.ndarray, B: int = 500, seed: int = 42) -> Tuple[float, Tuple[float, float]]:
    """AUC via rank statistic with bootstrap CI."""
    rng = np.random.default_rng(seed)
    def auc_once(sp, sn):
        npos, nneg = len(sp), len(sn)
        pair = 0.0
        ties = 0.0
        for s in sp:
            gt = (sn < s).sum()
            eq = (sn == s).sum()
            pair += gt
            ties += eq
        auc = (pair + 0.5 * ties) / (npos * nneg)
        return float(auc)
    auc_hat = auc_once(scores_pos, scores_neg)
    aucs = []
    for _ in range(B):
        sp = rng.choice(scores_pos, size=len(scores_pos), replace=True)
        sn = rng.choice(scores_neg, size=len(scores_neg), replace=True)
        aucs.append(auc_once(sp, sn))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return auc_hat, (float(lo), float(hi))

# ------------------------ LLM Judge ------------------------

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

def judge_prompt_classic(prompt_text: str, output_text: str) -> str:
    return f"""You are a precise classifier for AI assistant responses to conflicting instructions.

ORIGINAL PROMPT: {prompt_text}

ASSISTANT RESPONSE: {output_text}

Classify this response into exactly ONE of these 7 types:

Type 1 - Refusal-only: Declines to comply and does NOT ask a clarifying question. Provides no solution.
Type 2 - Refusal+Clarification: Flags the conflict/impossibility AND asks a targeted clarifying question. No solution yet.
Type 3 - Pick-one: Satisfies one constraint, violates the other.
Type 4 - Mix/compromise: Attempts to satisfy both by mixing the two constraints in a single answer; yields an incoherent hybrid.
Type 5 - Multiple outputs: Produces one answer per mutually exclusive constraint.
Type 6 - Normal Behavior: Correctly performs the task for a non-conflicting prompt without refusal or unnecessary clarification.
Type 7 - Other: Any other behavior not captured above.

TIEBREAK: Prefer the most specific type (e.g., Type 2 over Type 1 if a question is asked).

Output the number within <type> like <type>3</type> and a brief rationale within <rationale>...</rationale>.
"""

def judge_prompt_constraint(prompt_text: str, output_text: str) -> str:
    return f"""You are a precise classifier for AI assistant responses to prompts with TWO constraints (<ConstraintA> and <ConstraintB>).

Tasks:
1) Extract constraints A and B from the prompt.
2) Decide if the assistant produced ONE output or MULTIPLE outputs.
3) For a SINGLE output, evaluate:
   satisfies_A: true/false/uncertain
   satisfies_B: true/false/uncertain
   (Use surface evidence: commas vs tabs; JSON vs XML; snake_case vs camelCase; casing; code blocks; etc.)
4) Map to exactly ONE type:
   2 Refusal+Clarification; 1 Refusal-only; 5 Multiple outputs;
   3 Pick-one (exactly one satisfied); 4 Mix/compromise (attempts both/violates both with mixed cues);
   6 Normal (non-conflicting done correctly); 7 Other.

ORIGINAL PROMPT:
{prompt_text}

ASSISTANT RESPONSE:
{output_text}

OUTPUT (XML):
<constraints><A>...</A><B>...</B></constraints>
<outputs>single|multiple</outputs>
<satisfies><A>true|false|uncertain</A><B>true|false|uncertain</B></satisfies>
<type>1|2|3|4|5|6|7</type>
<rationale>1–2 sentences with evidence.</rationale>
"""

def parse_judge_xml(s: str) -> Tuple[int, str]:
    def _grab(tag):
        try:
            a = s.index(f"<{tag}>") + len(f"<{tag}>")
            b = s.index(f"</{tag}>")
            return s[a:b].strip()
        except ValueError:
            return ""
    t = _grab("type")
    r = _grab("rationale")
    try:
        tnum = int(t)
        if 1 <= tnum <= 7:
            return tnum, r
    except Exception:
        pass
    return 7, "Could not parse; defaulted to Type-7"

def judge_label(model_name: str, prompt_text: str, output_text: str, constraint_aware: bool=False) -> Tuple[int, str]:
    prompt = judge_prompt_constraint(prompt_text, output_text) if constraint_aware else judge_prompt_classic(prompt_text, output_text)
    msgs = [{"role": "user", "content": prompt}]
    comp = client.chat.completions.create(model=model_name, messages=msgs)
    content = comp.choices[0].message.content
    return parse_judge_xml(content)

def label_batch(model_name: str, items: List[Tuple[str, str]], constraint_aware: bool=False) -> List[int]:
    types = []
    for prompt_text, output_text in items:
        t, _ = judge_label(model_name, prompt_text, output_text, constraint_aware=constraint_aware)
        types.append(t)
    return types

# ------------------------ Model / Template / Decoding ------------------------

def load_model(model_path: str, device: str = "auto"):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    mdl.eval()
    return mdl, tok

def build_formatted_prompt(tokenizer, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        msgs = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def greedy_with_intervention(
    model, tokenizer, user_text: str,
    layer_index: int,
    add_vec: np.ndarray = None,
    ablate_vec: np.ndarray = None,
    alpha: float = 0.0,
    max_new_tokens: int = 128,
    take_logits_first3: bool = True,
) -> Tuple[str, List[np.ndarray]]:
    """
    Greedy decode with optional intervention at a given layer after prefill.
    Returns (decoded_text, logits_list_first3) as float32 numpy arrays (first 3 steps).
    """
    assert (add_vec is None) ^ (ablate_vec is None) or (add_vec is None and ablate_vec is None), \
        "Specify either add_vec or ablate_vec or none."

    device = next(model.parameters()).device
    fmt = build_formatted_prompt(tokenizer, user_text)
    input_ids = tokenizer.encode(fmt, return_tensors="pt").to(device)

    # Prepare vectors (normalize) on device
    v_add = v_abl = None
    if add_vec is not None:
        v_add = torch.from_numpy(add_vec.astype(np.float32)).to(device)
        v_add = v_add / (torch.norm(v_add) + 1e-8)
    if ablate_vec is not None:
        v_abl = torch.from_numpy(ablate_vec.astype(np.float32)).to(device)
        v_abl = v_abl / (torch.norm(v_abl) + 1e-8)

    first3_logits: List[np.ndarray] = []

    def hook_fn(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.size(1) == 1:
            if v_add is not None and alpha != 0.0:
                v_cast = v_add.to(hs.dtype)
                hs[:, -1, :] = hs[:, -1, :] + (hs.new_tensor(alpha) * v_cast)
            elif v_abl is not None:
                v_cast = v_abl.to(hs.dtype)
                proj = torch.einsum("bd,d->b", hs[:, -1, :], v_cast)
                hs[:, -1, :] = hs[:, -1, :] - proj.unsqueeze(-1) * v_cast
        return hs

    handle = model.model.layers[layer_index].register_forward_hook(hook_fn)

    try:
        # prefill
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        past = out.past_key_values

        generated = []
        for step in range(max_new_tokens):
            last_token = input_ids[:, -1:]
            with torch.no_grad():
                out = model(last_token, use_cache=True, past_key_values=past)
            past = out.past_key_values

            logits = out.logits[:, -1, :]  # [1, vocab]
            if take_logits_first3 and step < 3:
                first3_logits.append(
                    logits.detach().to(dtype=torch.float32, device="cpu").numpy().squeeze(0)
                )

            next_id = torch.argmax(logits, dim=-1)
            next_id_int = next_id.item()
            generated.append(next_id_int)
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
            if tokenizer.eos_token_id is not None and next_id_int == tokenizer.eos_token_id:
                break

        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        return decoded, first3_logits
    finally:
        handle.remove()

def top_p_sample_from_logits(logits: torch.Tensor, temperature: float, top_p: float, gen: Optional[torch.Generator]=None) -> int:
    logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = cumsum <= top_p
    if not torch.any(keep):
        keep[0] = True
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_idx[keep], sorted_probs[keep])
    filtered = filtered / (filtered.sum() + 1e-12)
    if gen is None:
        next_id = torch.multinomial(filtered, num_samples=1).item()
    else:
        next_id = torch.multinomial(filtered, num_samples=1, generator=gen).item()
    return next_id

def sample_with_intervention(
    model, tokenizer, user_text: str,
    layer_index: int,
    add_vec: np.ndarray = None,
    ablate_vec: np.ndarray = None,
    alpha: float = 0.0,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 200,
    take_logits_first3: bool = False,
    seed: Optional[int] = None,
) -> Tuple[str, List[np.ndarray]]:
    """
    Nucleus sampling (temp/top_p) with same intervention hook as greedy.
    Returns (decoded_text, logits_list_first3) where logits_list_first3 are float32 numpy arrays.
    """
    assert (add_vec is None) ^ (ablate_vec is None) or (add_vec is None and ablate_vec is None), \
        "Specify either add_vec or ablate_vec or none."

    device = next(model.parameters()).device
    fmt = build_formatted_prompt(tokenizer, user_text)
    input_ids = tokenizer.encode(fmt, return_tensors="pt").to(device)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    v_add = v_abl = None
    if add_vec is not None:
        v_add = torch.from_numpy(add_vec.astype(np.float32)).to(device)
        v_add = v_add / (torch.norm(v_add) + 1e-8)
    if ablate_vec is not None:
        v_abl = torch.from_numpy(ablate_vec.astype(np.float32)).to(device)
        v_abl = v_abl / (torch.norm(v_abl) + 1e-8)

    first3_logits: List[np.ndarray] = []

    def hook_fn(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.size(1) == 1:
            if v_add is not None and alpha != 0.0:
                v_cast = v_add.to(hs.dtype)
                hs[:, -1, :] = hs[:, -1, :] + (hs.new_tensor(alpha) * v_cast)
            elif v_abl is not None:
                v_cast = v_abl.to(hs.dtype)
                proj = torch.einsum("bd,d->b", hs[:, -1, :], v_cast)
                hs[:, -1, :] = hs[:, -1, :] - proj.unsqueeze(-1) * v_cast
        return hs

    handle = model.model.layers[layer_index].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        past = out.past_key_values

        generated: List[int] = []
        for step in range(max_new_tokens):
            last_token = input_ids[:, -1:]
            with torch.no_grad():
                out = model(last_token, use_cache=True, past_key_values=past)
            past = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)

            if take_logits_first3 and step < 3:
                first3_logits.append(logits.detach().to(dtype=torch.float32, device="cpu").numpy())

            next_id = top_p_sample_from_logits(logits, temperature=temperature, top_p=top_p, gen=gen)
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break

        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        return decoded, first3_logits
    finally:
        handle.remove()

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
    For each site key (layer_{mid|late}_t{1..3}), compute per-prompt Δ_p:
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
    """Compute mean KL (first 3 tokens) and Top-10 overlap between baseline and steered (greedy)."""
    kls = []
    overlaps = []

    for text in control_prompts:
        # baseline logits (first 3)
        _, base_logits_list = greedy_with_intervention(
            model, tokenizer, text, layer_index=0, add_vec=None, ablate_vec=None,
            alpha=0.0, max_new_tokens=4
        )
        # steered logits (first 3)
        _, steer_logits_list = greedy_with_intervention(
            model, tokenizer, text, layer_index=layer_index, add_vec=v_unit, ablate_vec=None,
            alpha=alpha, max_new_tokens=4
        )
        L = min(len(base_logits_list), len(steer_logits_list), 3)
        for t in range(L):
            b = base_logits_list[t].squeeze()
            s = steer_logits_list[t].squeeze()
            p = softmax_np(b)
            q = softmax_np(s)
            kls.append(kl_np(p, q))
            overlaps.append(topk_overlap_idx(b, s, k=10))
    mean_kl = float(np.mean(kls)) if kls else 0.0
    mean_top10 = float(np.mean(overlaps)) if overlaps else 0.0
    return mean_kl, mean_top10

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
        for pi, text in enumerate(val_prompts):
            for si in range(num_samples):
                seed = base_seed + pi * 1000 + si
                out, _ = sample_with_intervention(
                    model, tokenizer, text, layer_index=0, add_vec=None, ablate_vec=None,
                    alpha=0.0, temperature=temperature, top_p=top_p, max_new_tokens=200, seed=seed
                )
                pairs.append((text, out))

    labels = label_batch(judge_model, pairs, constraint_aware=constraint_aware)
    N = len(labels)
    t5 = sum(1 for t in labels if t == 5)
    return float(t5 / max(1, N)), float(1.0 - t5 / max(1, N)), N, labels

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
    ap.add_argument("--val_decode", default="sample", choices=["sample", "greedy"],
                    help="Decoding mode for validation prompts (default: sample).")
    ap.add_argument("--val_temperature", type=float, default=0.8)
    ap.add_argument("--val_top_p", type=float, default=0.95)
    ap.add_argument("--val_samples", type=int, default=5,
                    help="Number of draws per validation prompt when using sampling.")

    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    set_seed(args.seed)

    # Load model/tokenizer
    model, tokenizer = load_model(args.model_path, args.device)

    # Load train activations
    arrays, meta = load_train_acts(Path(args.acts_path), Path(args.meta_path))
    timesteps = meta.get("timesteps", [1, 2, 3])
    layer_indices = meta["layer_indices"]
    print(f"Loaded activations keys: {list(arrays.keys())[:4]}..., n={meta.get('n')}, d_model={meta.get('d_model')}")
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

    # Baseline validation rates (sampling or greedy per flags)
    base_t5, base_not5, N_val, base_labels = eval_validation_baseline(
        model, tokenizer, val_prompts, args.judge_model, args.constraint_aware,
        decode_mode=args.val_decode, temperature=args.val_temperature, top_p=args.val_top_p,
        num_samples=args.val_samples, base_seed=1234
    )
    print(f"Validation baseline ({args.val_decode}): Type-5 rate={base_t5:.3f}, ¬Type-5 rate={base_not5:.3f}, N={N_val}")

    # Evaluate candidates across alphas
    alphas = [float(x) for x in args.alphas.split(",")]
    sel_rows = []

    for cand in candidates:
        v = cand["v_unit"]
        lidx = cand["layer_index"]
        lname = cand["layer_name"]
        pos = cand["pos"]

        for a in alphas:
            # Validation (steered) — sampling/greedy per flags
            rate5_s, not5_s, N_s = eval_validation_rates(
                model, tokenizer, val_prompts, lidx, v, a, args.judge_model, args.constraint_aware,
                decode_mode=args.val_decode, temperature=args.val_temperature, top_p=args.val_top_p,
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
                "n_ctrl": len(ctrl_prompts)
            }
            sel_rows.append(row)
            print(f"[{lname}@t{pos} | α={a:.2f}] ΔT5={d_type5:+.3f} Δ¬T5={d_not5:+.3f} KL={mean_kl:.3f} Top10={mean_top10:.3f} J={J:+.3f}")

    # Select best at α=0.4
    rows_a04 = [r for r in sel_rows if abs(r["alpha"] - 0.4) < 1e-6]
    if not rows_a04:
        raise RuntimeError("No rows found for α=0.4")
    best = max(rows_a04, key=lambda r: r["J_multi"])
    print("\nWinner @ α=0.4:")
    print(json.dumps(best, indent=2))

    # Retrieve the winning candidate vector
    win_layer = best["layer_name"]
    win_pos = best["pos"]
    win_key = f"layer_{win_layer}_t{win_pos}"
    win_cand = None
    for c in candidates:
        if c["key"] == win_key:
            win_cand = c
            break
    assert win_cand is not None
    v_star = win_cand["v_unit"]
    l_star = best["layer_index"]
    pos_star = win_pos
    alpha_star = 0.4

    # Necessity/sufficiency sanity (greedy decoding for determinism):
    # Addition should decrease ¬Type-5; ablation should increase ¬Type-5.
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
        auc, (lo, hi) = 0.5, (0.5, 0.5)

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
                     "decode": args.val_decode, "samples_per_prompt": args.val_samples if args.val_decode=="sample" else 1},
        "addition": {"alpha": float(alpha_star), "not_type5_rate": float(not5_add), "delta_not_type5": float(delta_not5_add), "decode": "greedy"},
        "ablation": {"not_type5_rate": float(not5_abl), "delta_not_type5": float(delta_not5_abl), "decode": "greedy"},
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
          f"--val_decode {args.val_decode} --val_temperature {args.val_temperature} --val_top_p {args.val_top_p} --val_samples {args.val_samples} "
          f"--alphas {args.alphas} --out_dir {args.out_dir}"
    )

if __name__ == "__main__":
    main()
