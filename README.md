# Wait a Second... - Enforcing Refusal + Clarification Under Instruction Conflicts

Goal: find a single activation-space direction that increases Type-2 (refusal + clarifying
question) on conflicting-instruction prompts while minimizing normal behaviour drift; test basic
  cross-conflict type generalization (formatting ↔ word-count).

## Quick start
```bash
# Create remote env by cloning the base (main) env with preinstalled libs
conda create -y -n nanda_mats --clone main

# Activate the env
conda activate nanda_mats

# Install only the missing packages from requirements
pip install -r requirements.txt --upgrade-strategy only-if-needed
```

Hello world:
```bash
python -m src.utils --demo softmax
```