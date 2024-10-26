python ./eval/inference/plot_inefficient_attentionForVideo.py \
    --model_path /mnt/data/victor/projects/attention_allocation/Video-LLaVA/checkpoints/llava \
    --cache_dir "./cache_dir" \
    --video_dir "./eval/inference/sample_demo_13.mp4" \
    --output_path "./eval/inference/extracted_frames_llava" \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 2056 \
    --fast-v-attention-rank 400 \
    --fast-v-agg-layer 4 \
    --question "What part of the video shows the athlete's number on their back?" \
