"""Primary script for fusing MiniCPMV vision tower
with Qwen language model for a strong vision-language model.
TO DO: finish cleaning up
TO DO: actually document functions
TO DO: figure out how to adjust for training (and only do it to projector)
TO DO: weed out what needs to be weeded out
TO DO: create standalone class for the model

? TO DO: add support for image embedding within text ?
? TO DO: add support for multi-image ?

(suggestive order to treat said to-do items)
"""
# TO DO figure out which import are not used
import base64
import inspect
import io
import json
import os
import re
import sys
import torch
import typing

from torch import nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# TO DO need to remove this line to get rid of local dependencies,
# but needed for CPMV classes ˇˇˇˇˇˇˇˇˇˇ 
sys.path.insert(0, "/data3/jmarie/MiniCPM-V") 
from chat import MiniCPMVChat, img2base64

cpm = MiniCPMVChat('openbmb/MiniCPM-V-2')

# Qwen imports
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Projector definition 
class CPMQwenProjector(nn.Module):
    def __init__(self, input_size, output_size):
        # Keep or remove modular sizes ?
        # Useful to keep if either models ever change
        # embedding spatial dimensions. Also seems cleaner
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=output_size),
            nn.GELU()
        )

        # Xavier init weights
        # Is this even necessary ???
        modules_gen = self.proj.modules()
        next(modules_gen)
        for layer in modules_gen:
            for parameter in layer.parameters():
                if len(parameter.size()) <= 1:
                    nn.init.normal_(parameter)
                else:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, x):
        return self.proj(x)

projector = CPMQwenProjector(input_size=2304, output_size=896)

# ESSENTIAL, CORE function to make the fusion work
# TO DO create a seperate classe ineheriting from both MiniCPMV and Qwen
# in order to remove dependencies from provided models, their embedding spaces
# and to make the code cleaner overall
def get_image_embeds(model, tokenizer, input_str: str, image: str, sampling=True, max_inp_length=2048, **kwargs) -> torch.Tensor:
    """Get image embeddings from the image, within the context of the input string
    """
    # TO DO remove need to provide model and tokenizer
    # TO DO is the input_str argument necessary ?
    vision_hidden_states=None # TO DO find a way to remove this
    
    # format input just like model.chat
    image = img2base64(image)
    image = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
    msgs = [{'role': 'user', 'content': input_str}] 

    # ˇˇˇˇˇˇˇˇˇˇ TO DO Whole section to weed out
    # msgs to prompt
    prompt = ""
    for i, msg in enumerate(msgs):
        role = msg["role"]
        content = msg["content"]
        assert role in ["user", "assistant"]
        if i == 0:
            if image is None:
                images = []
            else:
                assert role == "user", "The role of first msg should be user"
                if model.config.slice_mode:
                    images, final_placeholder = model.get_slice_image_placeholder(
                        image, tokenizer
                    )
                    content = final_placeholder + "\n" + content
                else:
                    images = [image]
                    content = (
                        tokenizer.im_start
                        + tokenizer.unk_token * model.config.query_num
                        + tokenizer.im_end
                        + "\n"
                        + content
                    )
        prompt += "<用户>" if role == "user" else "<AI>"
        prompt += content
    prompt += "<AI>"
    final_input = prompt

    if sampling:
        generation_config = {
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.05
        }
    else:
        generation_config = {
            "num_beams": 3,
            "repetition_penalty": 1.2,
        }

    generation_config.update(
        (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
    )
    # End of section to weed out
        
    # do the same as in generate, but stop when image_bound and input_embeds are loaded
    # MAX_INP_LENGTH ?
    # TOKENIZER ?
    # MAX_NEW_TOKENS ?

    # TO DO actually stop when both things are loaded,
    # ˇˇˇˇˇˇˇˇˇˇ basically weed it all out as well
    data_list = [final_input]
    img_list = [images]
    
    bs = len(data_list)
    if img_list == None:
        img_list = [[] for i in range(bs)]
    assert bs == len(img_list)

    model_inputs = model._process_list(tokenizer, data_list, max_inp_length)
    
    if vision_hidden_states is None:
        pixel_values = []
        for i in range(bs):
            img_inps = []
            for img in img_list[i]:
                img_inps.append(model.transform(img).to(model.device))
            if img_inps:
                pixel_values.append(img_inps)
            else:
                pixel_values.append([])
        model_inputs["pixel_values"] = pixel_values
    else:
        model_inputs["vision_hidden_states"] = vision_hidden_states

    with torch.inference_mode():
        (
            model_inputs["inputs_embeds"],
            vision_hidden_states,
        ) = model.get_vllm_embedding(model_inputs)
    # End of section to weed out
    
    # recuperate model_inputs
    image_bound, inputs_embeds = model_inputs["image_bound"], model_inputs["inputs_embeds"]

    # segment out image embeds
    image_embeds = []
    print(image_bound, image_bound[0][0][0])
    image_embeds.append(inputs_embeds[0][image_bound[0][0][0]: image_bound[0][-1][-1]])
    
    # return cut-out image embeds
    return image_embeds

EOT_TOKEN_ID=151644 # Why is this even there??
def embed_mixed_modal(model, cpm_tokenizer, input_str: str, image: str, sampling=True) -> list[int]:
    """Function to embed mixed modal input using MiniCPMV embedding
    projected into Qwen embedding space
    """
    # MiniCPMV tokenization + getting visual tokens
    cpm_image_embeds = get_image_embeds(model, cpm_tokenizer, input_str, image)
    # print("---------------Image embeds ---------------")
    # print(image_embeds, image_embeds[0].size())

    projector = CPMQwenProjector(input_size=2304, output_size=896).to(model.device).bfloat16()
    
    qwen_image_embeds = []
    with torch.no_grad():
        for image_embed in cpm_image_embeds:
            qwen_image_embeds.append(projector(image_embed))
    # print("QWEN EMBEDS", qwen_embeds, qwen_embeds[0].size())

    # Qwen tokenization
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_str.replace("[image]", "<|end_of_text|>")}
    ]
    print(messages)
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    qwen_text_tokens = tokenizer([text], return_tensors="pt").to(device)
    print(qwen_text_tokens)
    print("------------ DECODED IMAGE -------------")
    # print(tokenizer.batch_decode(qwen.lm_head(qwen_image_embeds[0])))
    # print(tokenizer.batch_decode(qwe))
    qwen_text_embeds = qwen.get_input_embeddings()(qwen_text_tokens["input_ids"][0]) # text tokens embedded in a 896-dimensional space
    # print("QWEN TEXT EMBEDS", qwen_text_embeds[0].unsqueeze(0).size(), qwen_text_embeds[1:].size(), qwen_image_embeds[0].size())

    qwen_embeds = torch.cat((qwen_text_embeds[0].unsqueeze_(0), qwen_image_embeds[0], qwen_text_embeds[1:]))
    # print(qwen_text_embeds)
    # print(qwen_image_embeds)

    print("QWEN_EMBEDS", len(qwen_text_embeds), len(qwen_image_embeds[0]))
    return qwen_embeds


multimodal_embeds = embed_mixed_modal(cpm.model.model,
                  cpm.model.tokenizer,
                  "Describe the text I just gave you",
                  "/home/jmarie/flares/positive_img/0000.png")

generated_ids = qwen.generate(inputs_embeds=multimodal_embeds.unsqueeze_(0), max_new_tokens=512) # literally generated tokens
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
