# Coherence
printf 'Coherence results'
printf '\n'
python evaluate_score.py --input_fp results/gpt4_coh_detailed_openai.json --dimension coherence
printf '\n'

# consistency
printf 'Consistency results'
printf '\n'
python evaluate_score.py --input_fp results/gpt4_con_detailed_openai.json --dimension consistency
printf '\n'

# Fluency
printf 'Fluency results'
printf '\n'
python evaluate_score.py --input_fp results/gpt4_flu_detailed_openai.json --dimension fluency
printf '\n'

# Relevance
printf 'Relevance results'
printf '\n'
python evaluate_score.py --input_fp results/gpt4_rel_detailed_openai.json --dimension relevance