#!/usr/bin/env python3
"""
Make train/validation splits from a dev_*_conflicts.json file.

Two modes:
  1) Label-aware (recommended): pass --labels data/dev_labels_corrected.jsonl
     - A prompt goes to train if MIX rate >= --mix_threshold (default 0.5)
       * MIX is taken from aggregate == "MIX" if available, else type == 4
     - Remaining prompts go to validation
  2) Random split (fallback): omit --labels
     - Uses --val_ratio (default 0.30) to split prompts randomly

Outputs:
  data/splits/conflict_train.json
  data/splits/conflict_validation.json

Each file is a list of objects: {"prompt_id": "<ID>"}  (as in your examples)
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_conflicts(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Expect list of {id, family, text, sibling_id}
    return data

def compute_mix_rates(
    labels_path: Path,
    prompt_ids: List[str],
    min_samples_per_prompt: int = 1
) -> Dict[str, float]:
    """
    Compute MIX rate per prompt_id, using:
      - aggregate == "MIX" if present, else type == 4
    Returns {prompt_id: mix_rate} for prompts seen in labels AND in prompt_ids.
    """
    prompt_set = set(prompt_ids)
    num = {pid: 0 for pid in prompt_ids}
    mix = {pid: 0 for pid in prompt_ids}

    for row in read_jsonl(labels_path):
        pid = row.get("prompt_id")
        if pid not in prompt_set:
            continue
        num[pid] += 1

        # Prefer aggregate if present and equals "MIX"
        agg = row.get("aggregate")
        typ = row.get("type")
        if agg == "MIX":
            mix[pid] += 1
        elif typ == 4:
            mix[pid] += 1

    rates = {}
    for pid in prompt_ids:
        if num[pid] >= min_samples_per_prompt:
            rates[pid] = (mix[pid] / num[pid]) if num[pid] > 0 else 0.0
    return rates

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_conflicts", required=True,
                    help="Path to dev_*_conflicts.json (e.g., prompts/dev_f1_conflicts.json)")
    ap.add_argument("--labels", default=None,
                    help="Optional path to labels JSONL (e.g., data/dev_labels_corrected.jsonl)")
    ap.add_argument("--mix_threshold", type=float, default=0.5,
                    help="MIX rate threshold for train (label-aware mode)")
    ap.add_argument("--min_samples_per_prompt", type=int, default=1,
                    help="Minimum labeled samples per prompt to consider in label-aware mode")
    ap.add_argument("--val_ratio", type=float, default=0.30,
                    help="Validation ratio for random split when --labels not provided")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for random split")
    ap.add_argument("--out_dir", default="data/splits",
                    help="Output directory for conflict_train.json and conflict_validation.json")
    args = ap.parse_args()

    dev_conflicts_path = Path(args.dev_conflicts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conflicts = load_conflicts(dev_conflicts_path)
    prompt_ids = [c["id"] for c in conflicts]

    train_ids: List[str] = []
    val_ids: List[str] = []

    if args.labels:
        labels_path = Path(args.labels)
        print(f"Using label-aware split with labels: {labels_path}")
        rates = compute_mix_rates(labels_path, prompt_ids, args.min_samples_per_prompt)

        # Some prompts may be missing from rates (no labels or below min samples) → send to validation
        for pid in prompt_ids:
            r = rates.get(pid, None)
            if r is not None and r >= args.mix_threshold:
                train_ids.append(pid)
            else:
                val_ids.append(pid)

        print(f"Computed MIX rates for {len(rates)}/{len(prompt_ids)} prompts.")
        # Small safety: if train is empty, fallback to random split
        if len(train_ids) == 0:
            print("⚠️ No prompts met the threshold; falling back to random split.")
            random.seed(args.seed)
            shuffled = prompt_ids[:]
            random.shuffle(shuffled)
            cut = int(round((1.0 - args.val_ratio) * len(shuffled)))
            train_ids, val_ids = shuffled[:cut], shuffled[cut:]
    else:
        print("No labels provided. Using random split.")
        random.seed(args.seed)
        shuffled = prompt_ids[:]
        random.shuffle(shuffled)
        cut = int(round((1.0 - args.val_ratio) * len(shuffled)))
        train_ids, val_ids = shuffled[:cut], shuffled[cut:]

    # Write outputs in the requested format: list of {"prompt_id": "..."}
    train_out = [{"prompt_id": pid} for pid in sorted(train_ids)]
    val_out = [{"prompt_id": pid} for pid in sorted(val_ids)]

    train_path = out_dir / "conflict_train.json"
    val_path = out_dir / "conflict_validation.json"
    train_path.write_text(json.dumps(train_out, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_out, indent=2), encoding="utf-8")

    print(f"✅ Wrote {len(train_out)} train prompts → {train_path}")
    print(f"✅ Wrote {len(val_out)} validation prompts → {val_path}")

    if args.labels:
        # Quick summary of thresholding
        print(f"\nSummary (label-aware): mix_threshold={args.mix_threshold}, "
              f"min_samples_per_prompt={args.min_samples_per_prompt}")
    else:
        print(f"\nSummary (random): val_ratio={args.val_ratio}, seed={args.seed}")

if __name__ == "__main__":
    main()