run generate + label exp
python src/generate_and_label.py --prompt_string "write a one-sentence definition of photosynthesis.
Constraint A: English ONLY.
Constraint B: Italian ONLY.
Both constraints are equally important. Produce exactly one output." --save_output --save_logits --skip_spot_check --samples_per_prompt 10

run generate + label dev (full)
python src/generate_and_label.py --save_output --save_logits --skip_spot_check


random split
python src/make_splits.py  --dev_conflicts prompts/separable_constraints/dev_f1_conflicts.json  --val_ratio 0.3  --seed 42  --out_dir data/splits_across

labels aware split
python src/make_splits.py  --dev_conflicts prompts/separable_constraints/dev_f1_conflicts.json  --labels data/dev_labels.jsonl  --mix_threshold 0.5  --min_samples_per_prompt 10  --val_ratio 0.3  --seed 42  --out_dir data/splits_across


store activations
python src/acts.py  --model_path ./models/llama-3.1-8b-instruct  --device auto  --seed 42  --gens_path data/dev_gens.jsonl  --labels_path data/dev_labels.jsonl  --train_split data/splits_across/conflict_train.json  --output_dir data/acts/across


run intervention and select best direction
python src/vector_multi_across.py  --acts_path data/acts/across/train.npz  --meta_path data/acts/across/train_meta.json  --prompt_dir prompts/separable_constraints  --val_split data/splits_across/conflict_validation.json  --controls_split data/splits_across/controls_gold.json  --alphas 0.2,0.4  --out_dir artifacts




















non zero multiple answers:
F1_CONF_03
F1_CONF_04
F1_CONF_07
F1_CONF_11
F1_CONF_12
F1_CONF_13
F1_CONF_16
F2_SEED_02
F2_CONF_05
F2_CONF_09