from videollava import LlavaLlamaForCausalLM
from videollava.model.language_model.llava_llama import LlavaConfig

from transformers import AutoModelForCausalLM
import torch


pretrain_mm_mlp_adapter = "checkpoints/Video-LLaVA-Pretrain-7B/mm_projector.bin"
pretrain_mm_mlp_adapter = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
config = LlavaConfig.from_pretrained("LanguageBind/Video-LLaVA-7B")

# Video-Llava from scratch
backbone = LlavaLlamaForCausalLM(config)

# Video-Llava fine-tuned
finetuned = LlavaLlamaForCausalLM.from_pretrained("LanguageBind/Video-LLaVA-7B")

# Video-Llava fine-tuned from hf
#pretrained_hf = AutoModelForCausalLM.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

backbone.load_state_dict(pretrain_mm_mlp_adapter, strict=False)
backbone.save_pretrained("checkpoints/Video-Llava-Pretrain-7B")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("LanguageBind/Video-LLaVA-7B")
tokenizer.save_pretrained("checkpoints/Video-Llava-Pretrain-7B")

from transformers import AutoModel
vicuna = AutoModel.from_pretrained("lmsys/vicuna-7b-v1.5")
backbone.model.load_state_dict(vicuna.state_dict(),strict=False)
backbone.save_pretrained("checkpoints/Video-Llava-Pretrain-7B-Real")
tokenizer = AutoTokenizer.from_pretrained("LanguageBind/Video-LLaVA-7B")
tokenizer.save_pretrained("checkpoints/Video-Llava-Pretrain-7B-Real")
