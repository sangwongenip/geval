# Coherence
# python autogen_eval_debate.py --prompt_fp prompts/summeval/coh_detailed.txt --aspect Coherence --save_fp results/gpt4_coh_detailed_openai.json

# Relevance
python autogen_eval_debate.py --prompt_fp prompts/summeval/rel_detailed.txt --aspect Relevance --save_fp results/gpt4_rel_detailed_openai.json

# Fluency
python autogen_eval_debate.py --prompt_fp prompts/summeval/flu_detailed.txt --aspect Fluency --save_fp results/gpt4_flu_detailed_openai.json
