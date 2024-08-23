"""Primary script for fusing MiniCPMV vision tower with Qwen language model
for a strong vision-language model.
This is still very rough work.
~ TO DO: implement training capabilities, work on edge cases
TO DO: weed out what needs to be weeded out for embed_img/get_mm_emebds
TO DO: create standalone class for the complete MiniCPMV-Qwen model
TO DO: move projector to standalone file, move complete model to standalone file
       also containing training script ?

? FEATURE: enable image embedding contextually within text ?
? FEATURE: enable multi-image prompts ?

(suggestive order to treat said to-do items)

Someday in the future, the decoder will be fine-tuned in order to engrain
image-interpreting capabilities into the Qwen decoder
"""
import base64
import io
import json
import logging
import os
import random
import sys
import torch
import typing

from tqdm import tqdm
from torch import nn
from PIL import Image
from torch.nn.functional import cross_entropy
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Testing imports
import inspect
import traceback

# TO DO need to remove this line to get rid of local dependencies,
# but needed for CPMV classes ˇˇˇˇˇˇˇˇˇˇ
sys.path.insert(0, "/data3/jmarie/MiniCPM-V")
from chat import MiniCPMVChat, img2base64
cpm = MiniCPMVChat('openbmb/MiniCPM-V-2')
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

IMAGE_PREFIX_TOKENS = torch.tensor(tokenizer("An image containing ").data["input_ids"])
IMAGE_PREFIX_LEN = IMAGE_PREFIX_TOKENS.shape[-1]

# Projector definition
# Integrate to custom model class
class CPMQwenProjector(nn.Module):
    """Projector class for CPM vision embeddings to Qwen text embeddings projection.
    Default call is evaluation of randomly initialized weights, saved weights
    thus need to be loaded
    """
    def __init__(self, cpm_dim: int, qwen_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features=cpm_dim, out_features=qwen_dim)
        self.gelu = nn.GELU()
        self.device = device

        self.apply(self._init_projector_weights)
        self.eval()

        return

    def forward(self, x):
        projected_image = self.gelu(self.linear(x))
        projection_len = torch.norm(projected_image, dim=-1, keepdim=True)

        return projected_image/projection_len

    def load_projector_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

        return

    def save(self, path):
        torch.save(self.state_dict(), path)

        return

    def _init_projector_weights(self, module):
        # Initialize all weights assuming the NN only has linear layers
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            module.bias.data.fill_(0)
        return
