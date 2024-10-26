import math
import os
import argparse
import json

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import json
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
from datasets import load_from_disk, load_dataset
import torch
import json
from tqdm import tqdm
import re

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm

import csv

import pdb
import numpy as np
f = None


def visualize_attention(multihead_attention, output_path="atten_map_1.png", title="Layer 5", token_positions=None):
    # Average over the heads
    averaged_attention = torch.mean(multihead_attention, dim=1).float()  # Shape: (batch_size, total_tokens, total_tokens)
    
    # For batch_size=1
    averaged_attention = averaged_attention[0]  # Shape: (total_tokens, total_tokens)
    
    # Apply different pooling strides for different token types
    # We'll define pooling strides for each token type
    # For example:
    # System Prompt: stride 1 (no pooling)
    # Image Tokens: stride 16
    # Question Tokens: stride 1 (no pooling)
    # Answer Tokens: stride 1 (no pooling)
    
    # First, define the ranges for each token type
    if token_positions is not None:
        boundaries = [
            ('S', 0, token_positions['adjusted_image_token_start']),  # System Prompt
            ('I', token_positions['adjusted_image_token_start'], token_positions['adjusted_image_token_end']),  # Image Tokens
            ('Q', token_positions['adjusted_question_start'], token_positions['adjusted_question_end']),  # Question Tokens
            ('A', token_positions['adjusted_question_end'], averaged_attention.shape[0])  # Answer Tokens
        ]
    else:
        # If token_positions is not provided, treat the entire sequence as one type
        boundaries = [('All', 0, averaged_attention.shape[0])]
    
    # Define pooling strides for each token type
    pooling_strides = {
        'S': 1,   # No pooling for System Prompt
        'I': 16,  # Pooling with stride 16 for Image Tokens
        'Q': 1,   # No pooling for Question Tokens
        'A': 1    # No pooling for Answer Tokens
    }
    
    # Apply pooling separately to each block and then assemble the pooled blocks
    pooled_attention_blocks = []
    pooled_token_labels = []
    pooled_token_positions = {}
    current_position = 0
    
    for token_type, start, end in boundaries:
        stride = pooling_strides.get(token_type, 1)
        block = averaged_attention[start:end, start:end]
        
        if stride > 1:
            # Apply pooling to the block
            block = torch.nn.functional.avg_pool2d(
                block.unsqueeze(0).unsqueeze(0),
                kernel_size=stride,
                stride=stride
            ).squeeze(0).squeeze(0)
        
        # Update token positions for the current block
        pooled_block_size = block.shape[0]
        pooled_token_positions[f'{token_type}_start'] = current_position
        pooled_token_positions[f'{token_type}_end'] = current_position + pooled_block_size
        
        # Create labels for the pooled tokens
        pooled_token_labels.extend([token_type] * pooled_block_size)
        
        # Append the pooled block to the list
        pooled_attention_blocks.append(block)
        
        # Update the current position
        current_position += pooled_block_size
    
    # Assemble the pooled blocks into a full attention matrix
    pooled_attention = torch.block_diag(*pooled_attention_blocks)
    
    # Since attention can flow between blocks, we need to handle off-diagonal blocks
    # For simplicity, we can pool the off-diagonal blocks with appropriate strides
    total_pooled_tokens = pooled_attention.shape[0]
    full_pooled_attention = torch.zeros((total_pooled_tokens, total_pooled_tokens))
    
    for i, (type_i, start_i, end_i) in enumerate(boundaries):
        stride_i = pooling_strides.get(type_i, 1)
        for j, (type_j, start_j, end_j) in enumerate(boundaries):
            stride_j = pooling_strides.get(type_j, 1)
            block = averaged_attention[start_i:end_i, start_j:end_j]
            if stride_i > 1 or stride_j > 1:
                # Apply pooling to the block
                block = torch.nn.functional.avg_pool2d(
                    block.unsqueeze(0).unsqueeze(0),
                    kernel_size=(stride_i, stride_j),
                    stride=(stride_i, stride_j)
                ).squeeze(0).squeeze(0)
            # Place the block in the appropriate position
            pooled_start_i = pooled_token_positions[f'{type_i}_start']
            pooled_end_i = pooled_token_positions[f'{type_i}_end']
            pooled_start_j = pooled_token_positions[f'{type_j}_start']
            pooled_end_j = pooled_token_positions[f'{type_j}_end']
            full_pooled_attention[pooled_start_i:pooled_end_i, pooled_start_j:pooled_end_j] = block
    
    # Proceed to plot the full_pooled_attention
    # Plot the attention matrix
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(10, 10), dpi=400)  # Adjust figsize if needed
    
    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=full_pooled_attention.max().item())
    
    ax = sns.heatmap(full_pooled_attention,
                     cmap=cmap,
                     norm=log_norm,
                     cbar_kws={'label': 'Attention score'},
                     xticklabels=False,
                     yticklabels=False)
    
    # Set tick labels
    tick_step = max(1, total_pooled_tokens // 20)
    tick_locations = np.arange(0, total_pooled_tokens, tick_step)
    ax.set_xticks(tick_locations)
    ax.set_yticks(tick_locations)
    ax.set_xticklabels([pooled_token_labels[i] for i in tick_locations], rotation=90, fontsize=3)
    ax.set_yticklabels([pooled_token_labels[i] for i in tick_locations], fontsize=3)
    
    # Adjust tick labels
    plt.yticks(rotation=0)
    
    plt.title(title)
    
    # Draw line separators between token types
    for token_type in ['S', 'I', 'Q', 'A']:
        start = pooled_token_positions.get(f'{token_type}_start')
        end = pooled_token_positions.get(f'{token_type}_end')
        if start is not None and end is not None:
            # Draw horizontal and vertical lines at the start and end positions
            if start > 0:
                ax.axhline(y=start, color='white', linestyle='--', linewidth=0.5)
                ax.axvline(x=start, color='white', linestyle='--', linewidth=0.5)
            ax.axhline(y=end, color='white', linestyle='--', linewidth=0.5)
            ax.axvline(x=end, color='white', linestyle='--', linewidth=0.5)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    # Find top attentions
    top_attentions = []
    for row in full_pooled_attention:
        top_values, top_indices = torch.topk(row, 10)
        top_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_attentions.append(top_line)
    
    # Save top attentions to CSV
    with open(output_path.replace(".png", ".csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(top_attentions)
    
    return top_attentions, full_pooled_attention

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--question', help='Question to ask the model.', required=True)
    parser.add_argument('--output_path', help='Path to the output directory.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--use-fast-v", type=bool, default=False)
    parser.add_argument("--fast-v-sys-length", type=int, default=36)
    parser.add_argument("--fast-v-image-token-length", type=int, default=2056)
    parser.add_argument("--fast-v-attention-rank", type=int, default=256)
    parser.add_argument("--fast-v-agg-layer", type=int, default=2)
    parser.add_argument("--not_load_mm", action="store_true", default=False)
    parser.add_argument("--not_load_mm_proj", action="store_true", default=False)
    parser.add_argument("--not_load_llm", action="store_true", default=False)
    parser.add_argument('--mm_proj_path', default='', required=False)
    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    outputs_attention = []
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    # Find the positions of different token types
    image_token_id = IMAGE_TOKEN_INDEX
    input_ids_list = input_ids[0].tolist()

    # Positions of image tokens in input_ids
    image_token_positions = [i for i, token_id in enumerate(input_ids_list) if token_id == image_token_id]

    # System prompt ends before the first image token
    system_prompt_end = image_token_positions[0] if image_token_positions else len(input_ids_list)

    # Question tokens start after the last image token
    question_start = image_token_positions[-1] + 1 if image_token_positions else system_prompt_end

    # Question tokens end at the end of input_ids
    question_end = input_ids.shape[1]

    # Calculate adjusted positions
    num_image_tokens = len(image_token_positions)
    expansion_per_image_token = 257 - 1  # Since original image token is replaced by 257 tokens

    adjusted_image_token_start = system_prompt_end
    adjusted_image_token_end = adjusted_image_token_start + num_image_tokens * 257

    # Adjusted question start and end
    adjusted_question_start = adjusted_image_token_end + (question_start - system_prompt_end - num_image_tokens)
    adjusted_question_end = adjusted_question_start + (question_end - question_start)

    token_positions = {
        'adjusted_image_token_start': adjusted_image_token_start,
        'adjusted_image_token_end': adjusted_image_token_end,
        'adjusted_question_start': adjusted_question_start,
        'adjusted_question_end': adjusted_question_end,
    }

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    global f
    print(prompt, input_ids, len([video_tensor]), video_tensor.shape, file=f)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=[stopping_criteria]
            )
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids["sequences"][:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        print("Response:", outputs, file=f)

    outputs_attention.append(output_ids['attentions'])
    return outputs_attention, input_ids, outputs, token_positions  # Return token positions as well

def run_inference(args):
    """
    Run inference using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)

    load_mm = not args.not_load_mm
    load_mm_proj = not args.not_load_mm_proj
    load_llm = not args.not_load_llm
    print("Load settings:", load_mm, load_mm_proj, load_llm)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, load_mm=load_mm,
        load_mm_proj=load_mm_proj, load_llm=load_llm, mm_proj_path=args.mm_proj_path
    )
    model = model.to(args.device)

    if args.use_fast_v:
        model.config.use_fast_v = True
        model.config.fast_v_sys_length = args.fast_v_sys_length
        model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.config.fast_v_agg_layer = args.fast_v_agg_layer
        model.config.fast_v_inplace = False  # This is a tuple for some reason
        model.config.use_cache = True
    else:
        model.config.use_fast_v = False

    model.config.model_max_length = args.model_max_length

    print("Use fast_v:", model.config.use_fast_v)
    total_layers = model.config.num_hidden_layers

    # Get the question
    qs = args.question

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "attn_maps"), exist_ok=True)
    global f 
    f = open(os.path.join(output_path, "output.txt"), "w")

    temp_path = os.path.join(args.video_dir)
    if os.path.exists(temp_path):
        video_path = temp_path
        outputs_attention, input_ids, outputs, token_positions = get_model_output(
            model, processor['video'], tokenizer, video_path, qs, args
        )
    else:
        print(f"Video path {temp_path} does not exist.")
        return


    # Process attentions
    outputs_attention = outputs_attention[0]  # List of attentions per time step
    time_steps = len(outputs_attention)
    num_image_tokens = token_positions['adjusted_image_token_end'] // 257
    n_input = input_ids.shape[1] - num_image_tokens + num_image_tokens * 257  # Adjusted input length
    n_output = time_steps - 1
    total_tokens = n_input + n_output
    print("Total tokens:", total_tokens, file=f)
    batch_size = 1  # Assuming batch_size = 1
    num_heads = outputs_attention[0][0].shape[1]  # Number of heads

    # Initialize per layer, the full attention matrix
    full_attentions = []  # List of size total_layers

    for l in range(total_layers):
        # Initialize full attention matrix for layer l
        # Shape: (batch_size, num_heads, total_tokens, total_tokens)
        try:
            full_attention = torch.zeros(batch_size, num_heads, total_tokens, total_tokens, device=args.device)
        except:
            full_attention = torch.zeros(batch_size, num_heads, total_tokens, total_tokens, device="cpu")
        full_attentions.append(full_attention)

    # Now, for each layer
    for l in range(total_layers):
        for t in range(time_steps):
            attn = outputs_attention[t][l]  # Shape varies per t
            if t == 0:
                full_attentions[l][:, :, :n_input, :n_input] = attn
            else:
                q_pos = n_input + t - 1
                k_len = n_input + t
                full_attentions[l][:, :, q_pos:q_pos+1, :k_len] = attn

    # After constructing full attentions, visualize per layer
    for l in range(total_layers):
        attention = full_attentions[l].cpu()
        output_file = os.path.join(output_path, "attn_maps", f"atten_map_layer{l+1}.png")
        title = f"Layer {l+1}"
        top5_attention, average_attentions = visualize_attention(
            attention, output_path=output_file, title=title,
            token_positions=token_positions  # Pass token positions
        )

        # Convert attention scores to CPU and store in a list
    attention_scores_cpu = [layer_att.cpu() for layer_att in full_attentions]
    
    # Define the path to save the attention scores
    attention_scores_path = os.path.join(output_path, 'attention_scores.pt')

    # Save the attention scores using torch.save
    torch.save(attention_scores_cpu, attention_scores_path)
    f.close()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
