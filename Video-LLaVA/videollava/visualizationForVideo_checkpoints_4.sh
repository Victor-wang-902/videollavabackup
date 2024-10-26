export CUDA_VISIBLE_DEVICES=4


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-6000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-5000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

#echo 'After Finetune'
#python ./eval/inference/plot_inefficient_attentionForVideo.py \
##    --cache_dir "./cache_dir" \
#    --video_dir "./eval/inference/sample_demo_1.mp4" \
#    --output_path "./eval/inference/test" \
#    --fast-v-sys-length 36 \
#    --fast-v-image-token-length 2056 \
#    --fast-v-attention-rank 400 \
#    --fast-v-agg-layer 4 \
#    --question "Why is this video funny?" \

# python ./eval/inference/plot_inefficient_attentionForVideo.py \
#     --model_path /mnt/data/victor/projects/attention_allocation/Video-LLaVA/checkpoints/Video-Llava-Scratch-7B \
#     --cache_dir "./cache_dir" \
#     --video_dir "./eval/inference/sample_demo_13.mp4" \
#     --output_path "./eval/inference/extracted_frames_Finetune_scratch" \
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
#     --output_path "./eval/inference/extracted_frames_Finetune_scratch_2" \
#     --fast-v-sys-length 36 \
#     --fast-v-image-token-length 2056 \
#     --fast-v-attention-rank 400 \
#     --fast-v-agg-layer 4 \
#     --question "What is funny about the video?" \