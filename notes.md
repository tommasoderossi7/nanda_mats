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
python -m src.vector_multi_across_cleaned_save  --acts_path data/acts/across/train.npz  --meta_path data/acts/across/train_meta.json  --prompt_dir prompts/separable_constraints  --val_split data/splits_across/conflict_validation.json  --controls_split data/splits_across/controls_gold.json  --alphas 0.2,0.4  --out_dir artifacts

run intervention and select best direction (quick version to test end-to-end pipeline)
python -m src.vector_multi_across_cleaned_save_quick --alphas 0.3 --direction_selection_decode sample  --val_samples 2  --quick  --constraint_aware  --out_dir artifacts_quick  --intervention_dir data/interventions_quick


Next tasks:
- format steered model responses (generations and labels) as dev_gens.jsonl, dev_gens_cfg.json, dev_logits.jsonl, dev_labels_raw.jsonl, dev_labels.jsonl dev_label_stats.json files and place them inside the data/interventions folder
- modify vector_multi_across.py: move generation and labeling functions from the file to generate and label files






DATASET SIZE DETAILS --- From residual is mediated by a single direction paper:

    HARMFUL INSTRUCTIONS
    D_harmful_train --> 128 harmful instructions sampled from ADVBENCH, MALICIOUSINSTRUCT, TDC2023
    D_harmful_val --> 32 harmful instructions sampled from the HARMBENCH validation set (only "standard behaviours")
    Evaluation 3* --> 100 harmful instructions, spanning 10 categories from JAILBREAKBENCH
        ABLATION: "To bypass refusal, we perform directional ablation on the “refusal direction” ˆr, ablating it from
            activations at all layers and all token positions. With this intervention in place, we generate model
            completions over JAILBREAKBENCH, a dataset of 100 harmful instructions"
    Evaluation 4* --> 159 harmful instructions sampled from the HARMBENCH test set (only “standard behaviors”)
        WEIGHTS ORTHOGONALIZATION: to prevent the model to ever write in the residual stream to the refusal direction

    HARMLESS INSTRUCTIONS
    D_harmless_train --> 128 instructions sampled from ALPACA
    D_harmless_val --> 32 instructions sampled from ALPACA
    Evaluation 3* --> 100 instructions sampled from ALPACA
        ADDITION: "To induce refusal, we add the difference-in-means vector r to activations in layer l∗, the layer that
            the r was originally extracted from. We perform this intervention at all token positions. With
            this intervention in place, we generate model completions over 100 randomly sampled harmless
            instructions from ALPACA."


DATASET DETAILS FOR THIS PROJECT:

    >>> DIRECTION DISCOVERY
    TYPE-5 RESPONSES (conflict resolution behaviour type) 
        D_conf_type5_train --> X type-5 responses sampled from Y conflicting prompts
        D_conf_type5_val --> ~X/4 type-5 responses sampled from ~Y/4 conflicting prompts
    ¬TYPE-5 RESPONSES (conflict resolution behaviour type) 
        D_conf_nontype5_train --> X non type-5 responses sampled from Y conflicting prompts
        D_conf_nontype5_val --> ~X/4 non type-5 responses sampled from ~Y/4 conflicting prompts

    >>> DIRECTION EVALUATION
    ABLATION TEST (to reduce type5)
        Evaluation(test set) --> ~X responses sampled from ~Y held out conflicting prompts under ablation intervention (possibly different type of conflicts, eg tone conflict)    
    ADDITION TEST (to induce type5)
        Evaluation(test set) --> ~X responses sampled from ~Y held out conflicting prompts under addition intervention (possibly different type of conflicts, eg tone conflict)

    
    CONTROL SET OF NORMAL INSTRUCTIONS SAMPLED FROM ALPACA: (check how they controlled for behaviour drift in the refusal paper, I don't think they used any behaviour drift measure during direction discovery to chose the direction with maximal effect and minimal behaviour drift on a control set, but they measured behaviour drift only after the selection has been done, ie in eval/test phase)
    measure under addition and ablation KL divergence and topk tokens overlap w.r.t. the model that didn't receive any interventions





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