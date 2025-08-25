# Project Guardrails (Agent MUST comply)

## 1) Step gating
- The agent MUST NOT start a new step until the user writes exactly:
  `APPROVED: <step-id>`
- After each step, the agent MUST post:
  - `DONE: <step-id>`
  - CHANGELOG: list of files created/modified (paths) + a brief bullet per change
  - A RESULTS.md snippet containing:
    * Acceptance checks from PLAN.yaml for that step
    * The outputs/metrics/logs that satisfy those checks
- If acceptance is not met, STOP and ask for guidance.

## 2) Scope & safety
- Do not change previously approved function signatures without explicit approval.
- No external network calls for data or labeling. Use local/HF models only.
- Keep seeds fixed when requested and log all configs to JSON.
- No PII in logs or artifacts.

## 3) Metrics and thresholds (must report; starred items are enforced)
- Conflict outcomes: Type distribution (1..7) [report]
- Probe AUC at chosen site for T2 vs non-T2 [report]
- *ΔType-2 on conflicts* (target +15–30 pp at α≈0.4)
- *Benign drift on controls*:
  - Mean KL (first 3 tokens) ≤ 0.2 nats
  - Top-10 overlap ≥ 0.9
- Dose–response monotonicity violations ≤ 2% [report]
- Leave-one-family-out retained gain ≥ 50% [report]
- Necessity/sufficiency:
  - Addition along v* increases Type-2 (greedy eval)
  - Directional ablation along v* decreases Type-2 (greedy eval)

## 4) Reproducibility
- All scripts accept CLI args, write JSON/NPZ/PNG artifacts, and print an exact
  “REPRO CMD” to rerun the experiment (with prompt_id/sample_idx when relevant).
- Artifacts MUST include enough metadata to recreate runs.

## 5) Repo hygiene
- Keep code under `src/`, data under `data/`, derived artifacts under `artifacts/`,
  prompts under `prompts/`, and plots/notebooks under `notebooks/`.
- Avoid hard-coded paths; use relative paths rooted at repo top.
