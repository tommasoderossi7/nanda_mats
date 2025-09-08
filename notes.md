prompts to improve:
F1_CONF_04 --> provide something to be formatted (the llm often answers: "I'm ready what would you like to format?")


run generate + label exp
python src/generate_and_label.py --prompt_string "write a sentence about war" --save_output --save_logits --skip_spot_check --samples_per_prompt 2

run generate + label dev (full)
python src/generate_and_label.py --save_output --save_logits --skip_spot_check