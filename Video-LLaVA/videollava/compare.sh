echo 1
python compare_attention_sim.py --file1 eval/inference/before_pretrained_new/attention_scores.pt --file2 eval/inference/before_pretrained_new_2/attention_scores.pt --output_dir similarity --output before_pretrained_2
echo 2
python compare_attention_sim.py --file1 eval/inference/before_pretrained_new/attention_scores.pt --file2 eval/inference/before_pretrained_rerun/attention_scores.pt --output_dir similarity --output before_pretrained_rerun

echo 3
python compare_attention_sim.py --file1 eval/inference/pretrained_new/attention_scores.pt --file2 eval/inference/before_pretrained_new_2/attention_scores.pt --output_dir similarity --output pretrained_2

echo 4
python compare_attention_sim.py --file1 eval/inference/finetuned_new/attention_scores.pt --file2 eval/inference/finetuned_new_2/attention_scores.pt --output_dir similarity --output finetuned_2

echo 5
python compare_attention_sim.py --file1 eval/inference/from_scratch_new/attention_scores.pt --file2 eval/inference/from_scratch_new_2/attention_scores.pt --output_dir similarity --output from_scratch_2
echo 6
python compare_attention_sim.py --file1 eval/inference/from_scratch_new/attention_scores.pt --file2 eval/inference/from_scratch_rerun/attention_scores.pt --output_dir similarity --output from_scratch_rerun

echo 7
python compare_attention_sim.py --file1 eval/inference/llm/attention_scores.pt --file2 eval/inference/llm_2/attention_scores.pt --output_dir similarity --output llm

echo 8
python compare_attention_sim.py --file1 eval/inference/llm_new/attention_scores.pt --file2 eval/inference/llm_new_2/attention_scores.pt --output_dir similarity --output llm_new