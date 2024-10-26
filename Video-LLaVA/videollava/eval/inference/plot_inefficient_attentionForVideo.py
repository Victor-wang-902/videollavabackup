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
from datasets import load_from_disk,load_dataset
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

def visualize_attention(multihead_attention,output_path="atten_map_1.png",title="Layer 5"):
    #pdb.set_trace()
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    
    # pooling the attention scores  with stride 80
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)
    # averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 80, stride=80).squeeze(0).squeeze(0)
    
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(5, 5),dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    # set the x and y ticks to 20x of the original


    ax = sns.heatmap(averaged_attention,
                cmap=cmap,  # custom color map
                norm=log_norm,  # 
                # cbar_kws={'label': 'Attention score'},
                )
    
    # remove the x and y ticks
    
    # replace the x and y ticks with string

    x_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    y_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # change the x tinks font size
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    
    # make y label vertical
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)     
    
    plt.title(title)
    # tight layout
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)

        # Save top_five_line to the csv named output.csv
        with open(output_path.replace(".png",".csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(top_five_attentions)

        #print(top_five_line)
        
    return top_five_attentions,averaged_attention

# def reform_kvq_shape(kvq_trt):
#     pdb.set_trace()
#     if len(kvq_trt) == 3:
#         return kvq_trt[0]
#     nbs, nh, nseq, nfeat = kvq_trt.shape
#     kvq_trt = kvq_trt.transpose(1, 2).contiguous()
#     kvq_trt = kvq_trt.reshape(nbs, nseq, nh * nfeat)
#     return kvq_trt[0]

# def cal_cos_kvq(kvq_trts):
#     kvq_trt_cos = []
#     for idtk in range(1):
#         kvq_trt_cos.append([])
#         for idl in range(len(kvq_trts[idtk])):
#             cur_layer_kvq = reform_kvq_shape(kvq_trts[idtk][idl])
            
#             # print(cur_layer_kvq.shape)
#             cur_sims = []
#             for idt in range(len(cur_layer_kvq)):
#                 cur_sims.append(F.cosine_similarity(cur_layer_kvq[idt], cur_layer_kvq).unsqueeze(0))
#             cur_sims = torch.cat(cur_sims, dim = 0)
#             kvq_trt_cos[idtk].append(cur_sims.unsqueeze(0))
#             # layer_sims.append(cur_sims)
#         kvq_trt_cos[idtk] = torch.cat(kvq_trt_cos[idtk], dim = 0).unsqueeze(0)
#     # pdb.set_trace()
#     kvq_trt_cos = torch.cat(kvq_trt_cos, dim = 0)
    
#     return kvq_trt_cos

# def cal_cos_skip_value(kvq_trts):
#     kvq_trt_cos = []
#     for idtk in range(1):
#         kvq_trt_cos.append([])
#         for idl in range(len(kvq_trts[idtk])-1):
#             cur_layer_kvq = reform_kvq_shape(kvq_trts[idtk][idl])
#             next_layer_kvq = reform_kvq_shape(kvq_trts[idtk][idl+1])
#             cur_norm = torch.norm(cur_layer_kvq,dim=1)
#             # pdb.set_trace()
#             cur_sims = []
#             for idt in [0,12,27]:
#                 print(idt, cur_norm[idt].cpu().item(), cur_norm[399].cpu().item(), (cur_norm[idt]/cur_norm[399]).cpu().item())
#                 cur_sims.append(F.cosine_similarity(cur_layer_kvq[idt], next_layer_kvq[idt], dim=0).item())
#             # cur_sims = torch.cat(cur_sims, dim = 0)
#             kvq_trt_cos[idtk].append(cur_sims)
#             # layer_sims.append(cur_sims)
#         # kvq_trt_cos[idtk] = torch.cat(kvq_trt_cos[idtk], dim = 0).unsqueeze(0)
#     # pdb.set_trace()
#     # kvq_trt_cos = torch.cat(kvq_trt_cos, dim = 0)
    
#     return kvq_trt_cos

# def cal_scale_skip_value(kvq_trts):
#     kvq_trt_cos = []
#     for idtk in range(1):
#         kvq_trt_cos.append([])
#         for idl in range(len(kvq_trts[idtk])-1):
#             cur_layer_kvq = reform_kvq_shape(kvq_trts[idtk][idl])
#             next_layer_kvq = reform_kvq_shape(kvq_trts[idtk][idl+1])
#             # pdb.set_trace()
#             cur_sims = []
#             for idt in [0,12,27]:
#                 cur_layer_kvq_norm = torch.norm(cur_layer_kvq[idt], dim=0)
#                 next_layer_kvq_norm = torch.norm(next_layer_kvq[idt], dim=0)
#                 cur_sims.append((cur_layer_kvq_norm / next_layer_kvq_norm).item())
#             # cur_sims = torch.cat(cur_sims, dim = 0)
#             kvq_trt_cos[idtk].append(cur_sims)
#             # layer_sims.append(cur_sims)
#         # kvq_trt_cos[idtk] = torch.cat(kvq_trt_cos[idtk], dim = 0).unsqueeze(0)
#     # pdb.set_trace()
#     # kvq_trt_scale = torch.cat(kvq_trt_cos, dim = 0)
    
#     return kvq_trt_cos

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_path', help='Path to the output file.', required=True)
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
    # pdb.set_trace()
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

    #pdb.set_trace()
    #import inspect
    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    #video_tensor = video_processor.preprocess( return_tensors='pt')['pixel_values'][0].half().to(args.device)

    # pdb.set_trace()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=True,
            temperature=1.0,
            max_new_tokens=1024,
            use_cache=True,
            output_attentions=True,
            output_scores=True,
            #output_kqv = True,  
            return_dict_in_generate=True,
            stopping_criteria=[stopping_criteria]
            )
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids["sequences"][:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        print("response", outputs)


    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[:-len(stop_str)]
    # outputs = outputs.strip()
    
    #pdb.set_trace()
    ##values = output_ids.all_kqvs[0]
    # vnp_ls=[]
    # for idl in range(len(values)):
    #     value_np = values[idl][0].cpu().numpy()
    #     vnp_ls.append(value_np)
    # vnp_all = np.stack(vnp_ls, axis=0)
    # outputs_values.append(vnp_all)
    
    # Clear the file if it does exist
    ##if os.path.exists('./output_example/kq_values.npy'):
    ##    os.remove('./output_example/kq_values.npy')
    # Create the file if it doens't exist
    ##if not os.path.exists('./output_example'):
    ##    os.makedirs('./output_example')
    ##np.save('./output_example/kq_values.npy', values)
    
        
    # skip_value_sims = cal_cos_skip_value(output_ids.all_kqvs)
    # skip_value_scales = cal_scale_skip_value(output_ids.all_kqvs)
    # pdb.set_trace()

    # layer_sims = cal_cos_kvq(output_ids.all_kqvs)
    # cur_sim_folder = './output_example/kq_sims'
    # mean_coss = []
    # cur_part = 'feat' 
    # for idl in range(len(layer_sims[0])):
    #     cur_sim_np = layer_sims[0][idl].cpu().numpy()
    #     fig_name = '{}_cossim_{}.png'.format(cur_part, idl)
    #     fig_path = os.path.join(cur_sim_folder, fig_name)
    #     cur_title = '{} cosine sim {}'.format(cur_part, idl)
    #     draw_qks(fig_path, cur_sim_np, title=cur_title)
    #     mean_coss.append(np.mean(cur_sim_np).item())
    # fig_name = '{}_cossim_means.png'.format(cur_part)
    # fig_path = os.path.join(cur_sim_folder, fig_name)
    # draw_qks_mean(fig_path, np.array(mean_coss), '{} cosine sim values'.format(cur_part))
    # if output_kqv:
    #     layer_scales = cal_scale_kvq(output_ids.all_kqvs)
    #     cur_scale_folder = './output_example/kq_scales'
    #     mean_scales = []
    #     cur_part = 'feat'
    #     for idl in range(len(layer_scales[0])):
    #         cur_sim_np = layer_scales[0][idl].cpu().numpy()
    #         fig_name = '{}_scale_{}.png'.format(cur_part, idl)
    #         fig_path = os.path.join(cur_scale_folder, fig_name)
    #         cur_title = '{} scale sim {}'.format(cur_part, idl)
    #         draw_qks(fig_path, cur_sim_np, title=cur_title)
    #         mean_scales.append(np.mean(cur_sim_np).item())
    #     fig_name = '{}_scale_means.png'.format(cur_part)
    #     fig_path = os.path.join(cur_scale_folder, fig_name)
    #     draw_qks_mean(fig_path, np.array(mean_scales), '{} scale values'.format(cur_part))
    #     mean_scale = layer_scales.mean(dim=3)
    #     # for kid in kid_ls:
    
    outputs_attention.append(output_ids['attentions'])
    return outputs_attention


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)

    load_mm = not args.not_load_mm
    load_mm_proj = not args.not_load_mm_proj
    load_llm = not args.not_load_llm
    print(load_mm, load_mm_proj, load_llm)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_mm=load_mm, load_mm_proj=load_mm_proj, load_llm=load_llm, mm_proj_path=args.mm_proj_path)
    model = model.to(args.device)
    # pdb.set_trace()

    if args.use_fast_v == True:
        model.config.use_fast_v = True
        model.config.fast_v_sys_length = args.fast_v_sys_length
        model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.config.fast_v_agg_layer = args.fast_v_agg_layer
        model.config.fast_v_inplace = False, # This is a tuple for some reason
        model.config.use_cache = True
    else:
        model.config.use_fast_v = False
    #print(model.config)
    #raise Exception
    model.config.model_max_length = args.model_max_length
    # pdb.set_trace()
    print(model.config.use_fast_v)
    #model.model.reset_fastv()
    total_layers = model.config.num_hidden_layers

    # From args get the qs
    qs = args.question

    temp_path = os.path.join(args.video_dir)
    if os.path.exists(temp_path):
        video_path = temp_path
        # try:
        # Run inference on the video and add the output to the list
        outputs_attention = get_model_output(model, processor['video'], tokenizer, video_path, qs, args)

    output_path = args.output_path

    try:
        os.mkdir(output_path)
    except:
        pass

    try:
        os.mkdir(output_path+"/attn_maps")
    except:
        pass

    output_path = args.output_path
    # draw attention maps
    #pdb.set_trace()
    for i in outputs_attention:
        for j in range(0,total_layers):
            top5_attention,average_attentions = visualize_attention(i[0][j].cpu(),output_path=output_path+"/attn_maps/atten_map_"+str(j)+".png",title="Layer "+str(j+1))



if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
