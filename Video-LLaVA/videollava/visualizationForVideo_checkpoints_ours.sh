export CUDA_VISIBLE_DEVICES=4


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-250" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-500" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-750" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-1000" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-1000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \
    
    
echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-1250" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-1250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \
    
    
echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-1500" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-1500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-1750" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-1750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-2000" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-2000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-2250" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-2250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-2500" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-2500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-2750" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-2750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-3000" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-3000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-3250" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-3250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-3500" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-3500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-3750" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-3750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-4000" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-4000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-4250" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-4250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-4500" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-4500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-4750" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-4750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-5000" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-5000" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-5250" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-5250" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-5500" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-5500" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \

echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours/checkpoint-5750" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-5750" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "Why is this video funny?" \


echo 'After Finetune'
python ./eval/inference/plot_inefficient_attentionForVideo_test.py \
    --model_path "../checkpoints/videollava-7b-ours" \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_1.mp4" \
    --output_path "./eval/inference/finetuned-ours-6000" \
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