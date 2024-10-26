export CUDA_VISIBLE_DEVICES=4

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-250" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-500" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-750" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-1000" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-1000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \
    
    
echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-1250" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-1250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \
    
    
echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-1500" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-1500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-1750" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-1750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-2000" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-2000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-2250" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-2250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-2500" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-2500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-2750" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-2750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-3000" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-3000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-3250" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-3250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-3500" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-3500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-3750" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-3750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-4000" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-4000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-4250" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-4250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-4500" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-4500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Pretrain'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --mm_proj_path "../checkpoints/videollava-7b-pretrain/checkpoint-4750" \
    --model_base lmsys/vicuna-7b-v1.5 \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/pretrained-4750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


#echo 'After Finetune'
#python ./eval/inference/plot_inefficient_attentionForVideo.py \
#    --model_path LanguageBind/Video-LLaVA-7B \
#    --cache_dir "./cache_dir" \
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