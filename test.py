"""Primary script for fusing MiniCPMV vision tower with Qwen language model
for a strong vision-language model. Possible future developments (over this week or smth)
include renaming this file, for it to only be used for projector class definition and so on,
and moving the whole inference/training to a new file contaning the model class definition,
create scripts for training and for testing in different file.
This is still very rough, in-early-development work.  
~ TO DO: finish cleaning up, add type safety and work on edge cases for written functions
~ TO DO: complete projector class*
TO DO: figure out how to adjust for training (and only do it to projector) (goes with above)
TO DO: weed out what needs to be weeded out
TO DO: create standalone class for the model

? TO DO: add support for image embedding within text ?
? TO DO: add support for multi-image ?

* add hooks for projector class ?

(suggestive order to treat said to-do items)
"""
import base64
import io
import json
import os
import sys
import torch
import typing

from torch import nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Testing imports
import inspect

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
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct").to(device)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Projector definition
def init_weights(module):
    if type(module) == nn.Linear: # Careful, this is assuming Projector is only linear
        nn.init.xavier_normal_(module.weight.data)
        module.bias.data.fill_(0)
    return

class CPMQwenProjector(nn.Module):
    """Projector class for CPM vision embeddings to Qwen text embeddings projection.
    Default mode is initialization with random weights
    """
    def __init__(self, input_size, output_size):
        # Keep or remove modular sizes ? -> Keep, in case embedding spaces change (unlikely but you never know)
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=output_size),
            nn.GELU()
        )
        # ˇˇˇˇˇˇˇˇˇˇ Rethink whole section: do I want to load it or just initialize it? Depends on instant function of model -> Distinguish between new model and eval/training from checkpoint -> Randomly initialized mode will be the default, and load custom model within the training/inference script if necessary
        self.eval()
        self.apply(init_weights)
        
    def forward(self, x):
        # maybe revisit if structure made less compact
        return self.proj(x)

    def load_checkpoint(self, path):
        # model.load_state_dict(torch.load(path))
        return
    
    def save(self, path):
        # torch.save(self.state_dict(), path)
        return

projector = CPMQwenProjector(input_size=2304, output_size=896)

# ESSENTIAL, CORE function to make the fusion work
# TO DO create a seperat classe ineheriting from both MiniCPMV and Qwen
# in order to remove dependencies from provided models, their embedding spaces
# and to make the code cleaner overall
def get_image_embeds(model, tokenizer, input_str: str, image: str, sampling=True, max_inp_length=2048, **kwargs) -> torch.Tensor:
    """Get image embeddings from MiniCPMV's vision tower, including slice embeddings
    and other embeddings necessary for image context. (Maybe link a list of em?)
    Args:
        model: model from which vision tower will be used for embedding image
        tokenizer: tokenizer for vision-language model
        input_str: input string as provided by user
        image: ABSOLUTE file path to the image to be embedded. Web images are not supported, must be locally saved image
        sampling: ???
        max_input_length: ????
        kwargs: ??? Used in original MiniCPMV code but idk what use it has here
    
    Returns:
        `torch.Tensor` of shape (1, `len(image_embeds)`, `cpm_embedding_dim`):
            The image embeddings, with context included (split token embeddings, ...)
    """
    # is it right to retrieve the embedded tokens within the context of the sentence?
    # cuz the projector would theoretically have to somehow learn mapping the influence
    # of the sentence on the CPM embedding, and translate it to smth Qwen understands ?
    # Is that even the case? Is that even feasible?
    
    # TO DO remove need to provide model and tokenizer -> with standalone class
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
    image_embeds = inputs_embeds[0][image_bound[0][0][0]: image_bound[0][-1][-1]]) # for multiple image inference, replace image_embeds with a list to append to + change image_bound indices 
    
    # return cut-out image embeds
    return image_embeds

def embed_mixed_modal(model, cpm_tokenizer, input_str: str, image: str, sampling=True) -> list[int]:
    """Embed mixed-modal data to the Qwen embedding space. The data is embedded
    to the cpm multimodal space. The image embeds are picked out and subsequently 
    projected to the Qwen embedding space using a linear GELU unit.
    All inputs preparation for embedding is done here (formatting + prompting + tokenization)
    Args:
        model: model from which to extract visual embeds (MiniCPMV)
        cpm_tokenizer: tokenizer for `model`
        input_str: the complete string input from the user
        image: image absolute path as image input. Does NOT support web images
        sampling: idky it's there, used in CPM embedding

    Returns:
        `torch.Tensor` of shape (1, `len(input_ids)`, `qwen_embedding_space_dims`):
            Embeddings of the multimodal inputs, with the picture projected embeddings
            before the qwen text embeds
    """
    # IS IT RIGHT TO SHIFT THE QWEN EMBEDS ? maybe put padding tokens for embedding and then replace their embeddings with the projected picture's (the model that the text input is shifted and has smth before it)
    # MiniCPMV tokenization + getting visual tokens
    cpm_image_embeds = get_image_embeds(model, cpm_tokenizer, input_str, image)
    cpm_vision_embedding_dimension = image_embeds.size()[-1] # Change the element accessed for /multi-image 
    
    projector = CPMQwenProjector(input_size=2304, output_size=896).to(model.device).bfloat16() # TO DO moving to device and changing data type incorporated into class __init__ for completing /projector-class
    
    with torch.no_grad():
        qwen_image_embeds = projector(image_embed) # change qwen_image_embeds to list for /multi-image
    
    # Qwen tokenization
    # figure out how to shift tokens in this block
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_str}
    ]    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    qwen_text_tokens = tokenizer([text], return_tensors="pt").to(device) # figure out why text is passed as list -> for batch inference perhaps?
    qwen_text_embeds = qwen.get_input_embeddings()(qwen_text_tokens["input_ids"][0]) # text tokens embedded in a 896-dimensional space

    qwen_embeds = torch.cat((qwen_text_embeds[0].unsqueeze_(0), qwen_image_embeds[0], qwen_text_embeds[1:])) # careful! extracting only one column from a 2-D tensor return a 1-D Tensor (hence unsqueeze)

    return qwen_embeds


multimodal_embeds = embed_mixed_modal(cpm.model.model,
                  cpm.model.tokenizer,
                  "Describe the text I just gave you",
                  "/home/jmarie/flares/positive_img/0000.png")

generated_ids = qwen.generate(inputs_embeds=multimodal_embeds.unsqueeze_(0), max_new_tokens=512) # literally generated tokens
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
