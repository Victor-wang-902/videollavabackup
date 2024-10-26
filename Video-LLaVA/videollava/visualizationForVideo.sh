export CUDA_VISIBLE_DEVICES=0

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "What is funny about the video?" \
    #--load_mm False \
    #--question "Is the American flag visible in the video?" \


    # --use-fast-v True \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "/mnt/data/victor/projects/attention_allocation/Video-LLaVA/checkpoints/Video-LLaVA-Pretrain-7B" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "What is funny about the video?" \

echo 'before pretrain, model_path shouldnt matter'
python ./eval/inference/plot_inefficient_attentionForVideo.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/before_pretrained" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "What is funny about the video?" \
    --not_load_mm_proj

echo 'from scratch, model_path shouldnt matter'
python ./eval/inference/plot_inefficient_attentionForVideo.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/from_scratch" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "What is funny about the video?" \
    --not_load_mm_proj \
    --not_load_mm \
    --not_load_llm

# python ./eval/inference/plot_inefficient_attentionForVideo.py \
#     --model_path /mnt/data/victor/projects/attention_allocation/Video-LLaVA/checkpoints/Video-Llava-Scratch-7B \
#     --cache_dir "./cache_dir" \
#     --video_dir "./eval/inference/sample_demo_13.mp4" \
#     --output_path "./eval/inference/extracted_frames_pretrain_scratch" \
#     --fast-v-sys-length 36 \
#     --fast-v-image-token-length 2056 \
#     --fast-v-attention-rank 400 \
#     --fast-v-agg-layer 4 \
#     --question "What part of the video shows the athlete's number on their back?" \
#     #--question "Is the American flag visible in the video?" \


#     # --use-fast-v True \

# python ./eval/inference/plot_inefficient_attentionForVideo.py \
#     --model_path /mnt/data/victor/projects/attention_allocation/Video-LLaVA/checkpoints/Video-Llava-Scratch-7B \
#     --cache_dir "./cache_dir" \
#     --video_dir "./eval/inference/sample_demo_1.mp4" \
#     --output_path "./eval/inference/extracted_frames_pretrain_scratch_2" \
#     --fast-v-sys-length 36 \
#     --fast-v-image-token-length 2056 \
#     --fast-v-attention-rank 400 \
#     --fast-v-agg-layer 4 \
#     --question "What is funny about the video?" \