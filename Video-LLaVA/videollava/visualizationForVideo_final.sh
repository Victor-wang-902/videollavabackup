echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned_new" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "/projectnb/ivc-ml/vwang/projects/attention_allocation/Video-LLaVA/videollava/checkpoints/Video-LLaVA-Pretrain-7B" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained_new" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'before pretrain, model_path shouldnt matter'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/before_pretrained_new" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \
    --not_load_mm_proj

echo 'from scratch, model_path shouldnt matter'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --model_base "/projectnb/ivc-ml/vwang/projects/attention_allocation/Video-LLaVA/videollava/checkpoints/vicuna-7b-v1.5" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/from_scratch_new" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \
    --not_load_mm_proj \
    --not_load_mm \
    
    
echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_13.mp4" \
    --output_path "./eval/inference/finetuned_new_2" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Describe what is going on in the video." \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "/projectnb/ivc-ml/vwang/projects/attention_allocation/Video-LLaVA/videollava/checkpoints/Video-LLaVA-Pretrain-7B" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_13.mp4" \
    --output_path "./eval/inference/pretrained_new_2" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Describe what is going on in the video." \

echo 'before pretrain, model_path shouldnt matter'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_13.mp4" \
    --output_path "./eval/inference/before_pretrained_new_2" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Describe what is going on in the video." \
    --not_load_mm_proj

echo 'from scratch, model_path shouldnt matter'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --model_base "/projectnb/ivc-ml/vwang/projects/attention_allocation/Video-LLaVA/videollava/checkpoints/vicuna-7b-v1.5" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_13.mp4" \
    --output_path "./eval/inference/from_scratch_new_2" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Describe what is going on in the video." \
    --not_load_mm_proj \
    --not_load_mm \
    


echo 'before pretrain, model_path shouldnt matter'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/before_pretrained_rerun" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \
    --not_load_mm_proj

echo 'from scratch, model_path shouldnt matter'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --model_base "/projectnb/ivc-ml/vwang/projects/attention_allocation/Video-LLaVA/videollava/checkpoints/vicuna-7b-v1.5" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/from_scratch_rerun" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \
    --not_load_mm_proj \
    --not_load_mm \