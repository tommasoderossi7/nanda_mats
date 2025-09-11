"""
Label outputs into Types 1..7 using LLM-judge rubric via Openrouter API.
Output single best label per sample.

New: optional --constraint_aware flag
- extracts <ConstraintA> and <ConstraintB> from the prompt
- uses a constraint-aware judge prompt and parsing (but still maps to Types 1..7)
"""

import argparse
import json
import random
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_global_seed, write_jsonl  # write_jsonl used for final output

# --- optional tqdm for nicer progress ---
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False
    def tqdm(it, total=None, desc=None):
        # minimal fallback
        for i, x in enumerate(it, 1):
            if total:
                print(f"{desc or 'progress'}: {i}/{total}", end="\r")
            yield x
        if total:
            print(" " * 60, end="\r")

def _append_jsonl(handle, obj: dict):
    """Append a JSON line and fsync for durability."""
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    handle.write(line)
    handle.flush()
    os.fsync(handle.fileno())

def load_all_prompts(prompt_dir: Path) -> Dict[str, Dict]:
    """Loads all prompts and creates a sibling map."""
    all_prompts = {}
    sibling_map = {}
    for f in prompt_dir.glob("*.json"):
        with open(f, 'r', encoding="utf-8") as infile:
            data = json.load(infile)
            for p in data:
                all_prompts[p['id']] = p
                if 'sibling_id' in p:
                    sibling_map[p['id']] = p['sibling_id']
    return all_prompts, sibling_map


# Response type definitions from PLAN.yaml
# (Keep existing aggregates to avoid breaking downstream code.)
RESPONSE_TYPES = {
    1: {"name": "Type-1 Refusal-only", "rule": "Declines to comply and does NOT ask a clarifying question. No content solution.", "aggregate": "OTHER"},
    2: {"name": "Type-2 Refusal+Clarification (TARGET)", "rule": "Explicitly flags the conflict/impossibility AND asks targeted clarification question. Provides no content solution yet.", "aggregate": "OTHER"},
    3: {"name": "Type-3 Pick-one", "rule": "Satisfies one constraint, violates the other.", "aggregate": "PRIORITIZE"},
    4: {"name": "Type-4 Mix/compromise", "rule": "Attempts to satisfy both by mixing the two constraints in a single answer; yields an incoherent hybrid.", "aggregate": "MIX"},
    5: {"name": "Type-5 Multiple outputs", "rule": "Produces one answer per mutually exclusive constraint.", "aggregate": "OTHER"},
    6: {"name": "Type-6 Normal Behavior", "rule": "Correctly performs a non-conflicting task without refusal or unnecessary clarification.", "aggregate": "OTHER"},
    7: {"name": "Type-7 Other", "rule": "Any other behavior not captured above.", "aggregate": "OTHER"},
}

# ---------- Constraint handling helpers ----------

_CONSTRAINT_TAG_A = re.compile(r"<ConstraintA>\s*(.*?)\s*</ConstraintA>", re.IGNORECASE | re.DOTALL)
_CONSTRAINT_TAG_B = re.compile(r"<ConstraintB>\s*(.*?)\s*</ConstraintB>", re.IGNORECASE | re.DOTALL)

def extract_constraints_from_prompt(prompt_text: str) -> Dict[str, Optional[str]]:
    """Extract <ConstraintA> and <ConstraintB> contents from prompt text, if present."""
    A = None
    B = None
    mA = _CONSTRAINT_TAG_A.search(prompt_text or "")
    mB = _CONSTRAINT_TAG_B.search(prompt_text or "")
    if mA:
        A = mA.group(1).strip()
    if mB:
        B = mB.group(1).strip()
    return {"A": A, "B": B}

# ---------- Judge prompts (two modes) ----------

def create_llm_judge_prompt_classic(sample: Dict) -> str:
    """Original rubric prompt (classic mode)."""
    prompt_text = sample["text"]
    output_text = sample["output_text"]
    judge_prompt = f"""You are a precise classifier for AI assistant responses to conflicting instructions.

ORIGINAL PROMPT: {prompt_text}

ASSISTANT RESPONSE: {output_text}

Classify this response into exactly ONE of these 7 types:

Type 1 - Refusal-only: Declines to comply and does NOT ask a clarifying question. Provides no solution.

Type 2 - Refusal+Clarification (TARGET): Explicitly flags the conflict/impossibility AND asks a targeted clarifying question. Provides no content solution yet.

Type 3 - Pick-one: Satisfies one constraint, violates the other.

Type 4 - Mix/compromise: Attempts to satisfy both by mixing the two constraints in a single answer; yields an incoherent hybrid.

Type 5 - Multiple outputs: Produces one answer per mutually exclusive constraint.

Type 6 - Normal Behavior: Correctly performs the task for a non-conflicting prompt without refusal or unnecessary clarification.

Type 7 - Other: Any other behavior not captured above.

TIEBREAK RULE: If ambiguous, choose the most specific type (e.g., prefer Type 2 over Type 1 if a question is asked).

Output the number (1-7) within xml tags <type> like <type>3</type>, and a brief rationale (1-2 sentences) explaining your choice within xml tags <rationale> like <rationale>xyz</rationale>.
"""
    return judge_prompt

def create_llm_judge_prompt_constraint_aware(sample: Dict) -> str:
    """
    Constraint-aware rubric:
    - Explains that constraints are inside <ConstraintA> and <ConstraintB> in the prompt.
    - Asks the judge to extract & evaluate satisfies_A/B and map to one of the 7 types.
    - Uses structured XML output.
    """
    prompt_text = sample["text"]
    output_text = sample["output_text"]
    # Keep the constraints inside ORIGINAL PROMPT so the judge can see the tags directly.
    judge_prompt = f"""You are a precise classifier for AI assistant responses to prompts that contain TWO potentially conflicting constraints.

The ORIGINAL PROMPT includes two constraints marked with XML tags <ConstraintA> and <ConstraintB>. You must:
1) Extract the two constraints as plain text A and B.
2) Decide whether the assistant produced ONE single output or MULTIPLE outputs.
3) For a SINGLE output, assess each constraint:
   - satisfies_A: true/false/uncertain
   - satisfies_B: true/false/uncertain
   Use concrete surface evidence (commas vs newlines; code block markers ```; JSON braces vs XML angle brackets; snake_case vs camelCase; language tokens; casing; etc.).
4) Map to exactly ONE type:
   Type 2 (Refusal+Clarification): flags impossibility AND asks a targeted clarifying question, no solution.
   Type 1 (Refusal-only): refuses without a question; no solution.
   Type 5 (Multiple outputs): two-or-more separate answers for A and B.
   Type 3 (Pick-one): satisfies exactly one of A or B.
   Type 4 (Mix/compromise): attempts both within ONE output:
       - satisfies_A=true AND satisfies_B=true when A and B are mutually exclusive; OR
       - satisfies_A=false AND satisfies_B=false but the output shows features of BOTH A and B together.
   Type 6 (Normal behavior): prompt is non-conflicting and the task is correctly performed without refusal or unnecessary clarification.
   Type 7 (Other): anything else not captured above.
TIEBREAK: Prefer the most specific type supported by evidence (e.g., Type 2 over Type 1 if a clear question is asked).

ORIGINAL PROMPT:
{prompt_text}

ASSISTANT RESPONSE:
{output_text}

OUTPUT FORMAT (XML):
<constraints>
  <A>...</A>
  <B>...</B>
</constraints>
<outputs>single|multiple</outputs>
<satisfies>
  <A>true|false|uncertain</A>
  <B>true|false|uncertain</B>
</satisfies>
<type>1|2|3|4|5|6|7</type>
<rationale>1-2 sentences with concrete evidence.</rationale>
"""
    return judge_prompt

# ---------- Interventions functions -------------------------------

def judge_prompt_classic_type5(prompt_text: str, output_text: str) -> str:
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

def judge_prompt_constraint_type5(prompt_text: str, output_text: str) -> str:
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
    return 999, "Could not parse; defaulted to Type-999"

def judge_label(model_name: str, prompt_text: str, output_text: str, constraint_aware: bool=False) -> Tuple[int, str]:
    prompt = judge_prompt_constraint_type5(prompt_text, output_text) if constraint_aware else judge_prompt_classic_type5(prompt_text, output_text)
    msgs = [{"role": "user", "content": prompt}]
    comp = client.chat.completions.create(model=model_name, messages=msgs)
    content = comp.choices[0].message.content
    return parse_judge_xml(content)

def label_batch(model_name: str, items: List[Tuple[str, str]], constraint_aware: bool=False) -> List[int]:
    types = []
    for prompt_text, output_text in items:
        t, _ = judge_label(model_name, prompt_text, output_text, constraint_aware=constraint_aware)
        types.append(t)
        print(f"\n\nJudged Type-{t}\nPrompt: {prompt_text[:50]}\nResponse {output_text[:100]}.")
    return types

# ---------- Parsing judge output (supports both prompts) ----------

def _xml_get(text: str, tag: str) -> Optional[str]:
    """Safe XML-ish tag extractor for small judge outputs."""
    try:
        start = text.index(f"<{tag}>") + len(f"<{tag}>")
        end = text.index(f"</{tag}>")
        return text[start:end].strip()
    except ValueError:
        return None

def parse_llm_output(response: str) -> Optional[Dict[str, str]]:
    """
    Parse judge output. Supports:
    - classic: <type>..</type> and <rationale>..</rationale>
    - constraint-aware: same, plus optional <constraints>, <outputs>, <satisfies>
    Always returns at least {'type': int, 'rationale': str}; may include extras.
    """
    if not response or not isinstance(response, str):
        return None

    type_txt = _xml_get(response, "type")
    rationale = _xml_get(response, "rationale")

    if type_txt is None:
        return None

    try:
        label_type = int(type_txt.strip())
    except Exception:
        return None

    if not (1 <= label_type <= 7):
        return None

    parsed: Dict[str, str] = {"type": label_type, "rationale": rationale or ""}

    # Optional extra fields (constraint-aware mode)
    consA = _xml_get(response, "A")
    consB = _xml_get(response, "B")
    outputs = _xml_get(response, "outputs")
    satA = None
    satB = None
    # Satisfies tags are nested; we still try direct pull:
    satA = _xml_get(response, "A") if "<satisfies>" in response else None
    satB = _xml_get(response, "B") if "<satisfies>" in response else None

    # Avoid overwriting A/B from <constraints> with the <satisfies> section
    # We do a light disambiguation: if we saw <constraints>, keep those as constraint_* fields
    if "<constraints>" in response:
        parsed["constraint_A"] = consA or ""
        parsed["constraint_B"] = consB or ""

    if "<satisfies>" in response:
        # Extract satisfies A/B with a more specific regex if present
        satA_match = re.search(r"<satisfies>\s*.*?<A>(true|false|uncertain)</A>", response, re.IGNORECASE | re.DOTALL)
        satB_match = re.search(r"<satisfies>\s*.*?<B>(true|false|uncertain)</B>", response, re.IGNORECASE | re.DOTALL)
        if satA_match:
            parsed["satisfies_A"] = satA_match.group(1).lower()
        if satB_match:
            parsed["satisfies_B"] = satB_match.group(1).lower()

    if outputs is not None:
        parsed["outputs"] = outputs

    return parsed

# ---------- LLM classify ----------

def llm_judge_classify(openrouter_model: str, sample: Dict, constraint_aware: bool = False) -> Dict[str, str]:
    """Use LLM to classify the sample. Returns dict with 'type' and 'rationale' (+ optional extras)."""
    if constraint_aware:
        judge_prompt = create_llm_judge_prompt_constraint_aware(sample)
    else:
        judge_prompt = create_llm_judge_prompt_classic(sample)

    messages = [{"role": "user", "content": judge_prompt}]
    completion = client.chat.completions.create(
        extra_body={},
        model=openrouter_model,
        messages=messages
    )
    response = completion.choices[0].message.content
    parsed = parse_llm_output(response)
    if parsed is not None:
        return parsed
    return {"type": 999, "rationale": "Could not parse LLM output"}

# ---------- Manual spot check ----------

def manual_spot_check(labels: List[Dict], percent: float = 0.15) -> List[Dict]:
    """Interactive spot-check interface for manual label correction."""
    label_groups = defaultdict(list)
    for i, label_data in enumerate(labels):
        label_groups[label_data["type"]].append((i, label_data))

    to_check = []
    for _, items in label_groups.items():
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

        print(f"\n📝 FULL PROMPT:\n{label_data.get('text', 'N/A')}")
        print(f"\n🤖 FULL ASSISTANT RESPONSE:\n{label_data.get('output_text', 'N/A')}")

        print(f"\n🏷️  LLM JUDGE CLASSIFICATION:")
        print(f"Assigned Type: {current_label}")
        print(f"Type Name: {RESPONSE_TYPES.get(current_label, {'name':'N/A'})['name']}")
        print(f"Type Rule: {RESPONSE_TYPES.get(current_label, {'rule':'N/A'})['rule']}")
        print(f"Rationale: {label_data.get('rationale', 'N/A')}")
        print(f"Aggregate: {RESPONSE_TYPES.get(current_label, {'aggregate':'N/A'})['aggregate']}")

        # If available, show constraint-aware extras
        extra_bits = []
        if "constraint_A" in label_data or "constraint_B" in label_data:
            extra_bits.append(f"Constraint A: {label_data.get('constraint_A','')}")
            extra_bits.append(f"Constraint B: {label_data.get('constraint_B','')}")
        if "satisfies_A" in label_data or "satisfies_B" in label_data:
            extra_bits.append(f"satisfies_A: {label_data.get('satisfies_A','')}")
            extra_bits.append(f"satisfies_B: {label_data.get('satisfies_B','')}")
        if "outputs" in label_data:
            extra_bits.append(f"outputs: {label_data.get('outputs','')}")
        if extra_bits:
            print("\n🔧 Constraint-aware details:")
            for eb in extra_bits:
                print(f"  - {eb}")

        print(f"\n📋 ALL CLASSIFICATION OPTIONS:")
        for type_id in sorted(RESPONSE_TYPES.keys()):
            type_info = RESPONSE_TYPES[type_id]
            marker = "👉 " if type_id == current_label else "   "
            print(f"{marker}{type_id}: {type_info['name']}")
            print(f"      Rule: {type_info['rule']}")

        print(f"\n⚙️  ACTIONS:")
        print("  [Enter]: Keep current label")
        print("  1-7: Change to that type")
        print("  s: Skip remaining samples")
        print("  d: Show detailed notes")

        while True:
            choice = input(f"\nYour choice (1-7, Enter=keep, s=skip, d=details): ").strip()

            if choice == "":
                break
            elif choice.lower() == "s":
                print("Skipping remaining spot-checks...")
                return labels
            elif choice.lower() == "d":
                print(f"\n🔍 Tiebreak rule: If ambiguous, choose the most specific type (e.g., prefer Type 2 over Type 1).")
                continue
            elif choice.isdigit() and 1 <= int(choice) <= 7:
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
                print("❌ Invalid input. Please enter 1-7, Enter, 's', or 'd'")

    if corrections:
        print(f"\n✏️  Applied {len(corrections)} manual corrections")
        for corr in corrections:
            print(f"  {corr['sample_id']}: {corr['old_label']} → {corr['new_label']}")
    else:
        print(f"\n✅ No corrections needed")

    return labels


def main():
    parser = argparse.ArgumentParser(description="Label outputs into Types 1..7")
    parser.add_argument("--gens_to_label", default="dev",
                       help="generations split to label (dev/exp/test)")
    parser.add_argument("--prompt_dir", default="prompts",
                       help="Directory containing prompt JSON files")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3.1:free",
                       help="Openrouter model to use for LLM judge")
    parser.add_argument("--responses_to_label", default="all",
                       help="Number of responses to label (or 'all')")
    parser.add_argument("--save_output", action="store_true",
                       help="Save labeled outputs to file (enables progressive raw saves)")
    parser.add_argument("--spot_check_rate", type=float, default=0.15,
                       help="Fraction of labels to manually spot-check")
    parser.add_argument("--skip_spot_check", action="store_true",
                       help="Skip manual spot-checking")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # NEW: constraint-aware switch
    parser.add_argument("--constraint_aware", action="store_true",
                       help="Use constraint-aware judge (extract ConstraintA/B, evaluate satisfies_A/B, map to Types)")

    args = parser.parse_args()

    # Seed for stable subsampling if used
    set_global_seed(args.seed)

    # Paths
    tag = args.gens_to_label  # dev/exp/test
    inputs = f"data/{tag}_gens.jsonl"
    raw_labels_output = f"data/{tag}_labels_raw.jsonl"   # progressive, appended
    final_labels_output = f"data/{tag}_labels.jsonl"     # final, cleaned
    stats_output = f"data/{tag}_label_stats.json"

    # Load prompts (sibling map not used further but kept for parity)
    print(f"📂 Loading prompts from {args.prompt_dir}")
    all_prompts, sibling_map = load_all_prompts(Path(args.prompt_dir))
    print(f"Loaded {len(all_prompts)} prompts and created sibling map for {len(sibling_map)} conflicts.")

    # Load generated samples
    print(f"📂 Loading samples from {inputs}")
    with open(inputs, 'r', encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    if args.responses_to_label != "all":
        num_to_label = int(args.responses_to_label)
        samples = random.sample(samples, min(num_to_label, len(samples)))

    total = len(samples)
    print(f"Loaded {total} samples")

    # Prepare progressive writer (raw labels)
    raw_handle = None
    if args.save_output:
        Path(raw_labels_output).parent.mkdir(parents=True, exist_ok=True)
        raw_handle = Path(raw_labels_output).open("a", encoding="utf-8")

    # Classify
    print(f"\n🏷️  Classifying samples...  (constraint_aware={args.constraint_aware})")
    labels = []
    iterator = tqdm(enumerate(samples), total=total, desc="Labeling") if HAVE_TQDM else enumerate(samples)

    try:
        for i, sample in iterator:
            # Create a view that ensures we pass the full prompt text (with tags) & output
            label = llm_judge_classify(
                openrouter_model=args.model,
                sample=sample,
                constraint_aware=args.constraint_aware
            )

            raw_label_entry = {
                "prompt_id": sample["prompt_id"],
                "sample_idx": sample["sample_idx"],
                "type": label["type"],
                "rationale": label.get("rationale", ""),
                "text": sample.get("text"),
                "output_text": sample.get("output_text")
            }

            # Include optional constraint-aware extras if present
            for k in ("constraint_A", "constraint_B", "satisfies_A", "satisfies_B", "outputs"):
                if k in label:
                    raw_label_entry[k] = label[k]

            # Progressive append of raw record
            if raw_handle is not None:
                _append_jsonl(raw_handle, raw_label_entry)

            labels.append(raw_label_entry)

            # lightweight progress print every 5 items if no tqdm
            if not HAVE_TQDM and (i + 1) % 5 == 0:
                print(f"Progress: {i+1}/{total}")

    finally:
        if raw_handle is not None:
            raw_handle.close()

    print(f"✅ Classified {len(labels)} samples")

    # Optional manual spot-check (on in-memory labels list)
    if not args.skip_spot_check and len(labels) > 0:
        labels = manual_spot_check(labels, args.spot_check_rate)

    # Clean final labels (strip prompt/response text, keep type/agg/rationale)
    final_labels = [{
        "prompt_id": ld["prompt_id"],
        "sample_idx": ld["sample_idx"],
        "type": ld["type"],
        "aggregate": RESPONSE_TYPES.get(ld["type"], {"aggregate":"OTHER"})["aggregate"],
        "rationale": ld.get("rationale", "")
    } for ld in labels]

    # Stats
    label_counts = Counter(ld["type"] for ld in final_labels)
    total_final = len(final_labels)
    if total_final == 0:
        print("\n⚠️ No final labels to summarize.")
        return

    stats = {
        "total_samples": total_final,
        "total_prompts": len(set(ld["prompt_id"] for ld in final_labels)),
        "label_distribution_by_type": {
            k: f"{v} ({(v/total_final*100):.2f}%)" for k, v in label_counts.items()
        },
        "label_distribution_by_aggregate": {
            agg: f"{count} ({(count/total_final*100):.2f}%)"
            for agg, count in Counter(RESPONSE_TYPES.get(ld["type"], {"aggregate":"OTHER"})["aggregate"]
                                      for ld in final_labels).items()
        },
        "label_distribution_by_prompt": {
            p_id: {
                r_type: f"{cnt} ({(cnt/max(1,len([x for x in final_labels if x['prompt_id']==p_id]))*100):.2f}%) "
                        f"{RESPONSE_TYPES.get(r_type, {'aggregate':'OTHER'})['aggregate']}"
                for r_type, cnt in Counter(ld["type"] for ld in final_labels if ld["prompt_id"] == p_id).items()
            } for p_id in set(ld["prompt_id"] for ld in final_labels)
        },
        "type_names": {k: {"type": v["name"], "agg": v["aggregate"]} for k, v in RESPONSE_TYPES.items()},
        "constraint_aware": args.constraint_aware
    }

    # Save outputs
    print(f"\n💾 Saving outputs...")
    if args.save_output:
        Path(final_labels_output).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(Path(final_labels_output), final_labels)
        print(f"Final labels saved to: {final_labels_output}")

        Path(stats_output).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_output, 'w', encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_output}")
    else:
        print("Note: --save_output not set; outputs only kept in memory / stdout.")

    # Summary print
    print(f"\n📈 Label Distribution (all samples):")
    for type_id in sorted(label_counts.keys()):
        count = label_counts[type_id]
        name = RESPONSE_TYPES.get(type_id, {"name":"N/A"})["name"]
        print(f"  Type-{type_id} ({name}): {count} ({count/total_final*100:.1f}%)")

    print(f"\n🔄 REPRO CMD:")
    repro_cmd = f"python src/label.py --gens_to_label {tag} --model {args.model} --seed {args.seed}"
    if args.constraint_aware:
        repro_cmd += " --constraint_aware"
    if args.skip_spot_check:
        repro_cmd += " --skip_spot_check"
    if args.responses_to_label != 'all':
        repro_cmd += f" --responses_to_label {args.responses_to_label}"
    if args.save_output:
        repro_cmd += " --save_output"
    print(f"  {repro_cmd}")


if __name__ == "__main__":
    main()
