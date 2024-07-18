
"""Primary script for fusing MiniCPMV vision tower with Qwen language model
for a strong vision-language model. Future ideas include only using this file
for the projector class, and moving inference/training to a new file
for the the model class. The same applies for inference, training and testing scripts.
This is still very rough work.  
~ TO DO: implement training capabilities, work on edge cases
TO DO: weed out what needs to be weeded out
TO DO: create custom fusion class

? FEATURE: add support for image embedding within text ?
? FEATURE: add support for multi-image ?

(suggestive order to treat said to-do items)
"""
import base64
import io
import json
import os
import sys
import torch
import torch.nn.functional as F
import typing

from torch import nn
from PIL import Image
from torch.nn.modules.loss import _Loss
from torch.nn.function import cross_entropy
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Testing imports
import inspect

# TO DO need to remove this line to get rid of local dependencies,
# but needed for CPMV classes ˇˇˇˇˇˇˇˇˇˇ 
sys.path.insert(0, "/data3/jmarie/MiniCPM-V") 
from chat import MiniCPMVChat, img2base64

cpm = MiniCPMVChat('openbmb/MiniCPM-V-2')
# print(os.path.abspath(inspect.getfile(type(cpm.model.model))))
# TO DO: remove ?
device = "cuda" # the device to load the model onto

qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# Projector definition
# Integrate to custom model class
class CPMQwenProjector(nn.Module):
    """Projector class for CPM vision embeddings to Qwen text embeddings projection.
    Default mode is initialization with random weights
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size),
        self.gelu = nn.GELU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.apply(init_projector_weights)
        self.eval()
        
        return
        
    def forward(self, x):
        batch_size = x.size(dim=0)
        projected_image = self.gelu(torch.stack([self.linear for idx in batch_size])(x))
        projection_len = torch.norm(projected_image, dim=-1, keepdim=True)
        
        return projected_image/projection_len

    def load_projector_checkpoint(self, path):
        model.load_state_dict(torch.load(path))

        return
    
    def save_projector(self, path):
        torch.save(self.state_dict(), path)

        return
    
    def init_projector_weights(self):
        # Initialize all weights assuming the NN only has linear layers
        if isinstance(self.module, nn.Linear):
            nn.init.xavier_normal_(self.module.weight.data)
            self.module.bias.data.fill_(0)
        return

projector = CPMQwenProjector(input_size=2304, output_size=896).to(qwen.device).bfloat16() # TO DO moving to device and changing data type incorporated into class __init__ for completing /projector-class # fixfixfixfixfixfixfix
    

# Integrate to custom model class
def embed_image(mm_model,
                mm_tokenizer,
                input_str: str,
                image: str,
                max_inp_length: int=2048,
                **kwargs) -> torch.Tensor:
    """Get image embeddings from MiniCPMV's vision tower, including slice embeddings
    and other embeddings necessary for image context. (Maybe link a list of em?)
    Args:
        mm_model: multimodal model which will be used for embedding image
        mm_tokenizer: multimodal model's tokenizer
        input_str: input string as provided by user
        image: ABSOLUTE file path to the image to be embedded. Web images are not supported, must be locally saved image
        max_input_length: ????
        kwargs: ??? Used in original MiniCPMV code but idk what use they have here
    
    Returns:
        `torch.Tensor` of shape (1, `len(image_embeds)`, `cpm_embedding_dim`):
            The image embeddings, with context included (split token embeddings, ...)
    """
    # Integrate this function to standalone model class
    # Rethink using embeddings within the context of the sentence (future future)
    # (can it learn the influence of the position of the image in the sentence?)

    model = mm_model
    tokenizer = mm_tokenizer
    vision_hidden_states=None # Necessary?
    sampling = True # Review use of sampling
    
    # Copy paste from MiniCPM inference
    image = img2base64(image)
    image = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
    msgs = [{'role': 'user', 'content': input_str}] 

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

    # Maybe remove for this use case
    generation_config.update(
        (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
    )
        
    # MAX_INP_LENGTH ?
    # TOKENIZER ?
    # MAX_NEW_TOKENS ?

    # Copy paste from MiniCPMV generate func
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
    
    # Recuperate embedded model_inputs, segment out image embeds
    image_bound, inputs_embeds = model_inputs["image_bound"], model_inputs["inputs_embeds"]
    image_embeds = inputs_embeds[0][image_bound[0][0][0]: image_bound[0][-1][-1]] # for multiple image inference, replace image_embeds with a list to append to + change image_bound indices 
    
    return image_embeds

# Integrate to custommodel class
def embed_mixed_modal(mm_model,
                      mm_tokenizer,
                      lm_model,
                      lm_tokenizer,
                      projector,
                      input_str: str,
                      image: str,
                      sampling=True) -> list[int]:
    """Embed mixed-modal data to the Qwen embedding space. The data is embedded
    to the cpm multimodal space. The image embeds are picked out and subsequently 
    projected to the Qwen embedding space using a linear GELU unit.
    All inputs preparation for embedding is done here (formatting + prompting + tokenization)
    Args:
        mm_model: model from which to extract visual embeds (MiniCPMV)
        mm_tokenizer: tokenizer for multimodal model
        lm_model: language model used for generation
        lm_tokenizer: tokenizer for said language model
        projector: projector layer used for translation mm embeds to lm embeds
        input_str: the complete string input from the user
        image: image absolute path as image input. Does NOT support web images
        sampling: idky, it's there, and it's used in CPM embedding

    Returns:
        `torch.Tensor` of shape (1, `len(input_ids)`, `qwen_embedding_space_dims`):
            Embeddings of the multimodal inputs, with the picture projected embeddings
            before the qwen text embeds
    """
    # Maybe shift Qwen inputs prior to embedding? Would help 
    # MiniCPMV tokenization + getting visual tokens
    mm_image_embeds = embed_image(mm_model, mm_tokenizer, input_str, image)
    mm_vision_embedding_dimension = mm_image_embeds.size()[-1] # Change the element accessed for /multi-image 

    # torch.no_grad really ?
    with torch.no_grad():
        lm_image_embeds = projector(mm_image_embeds) # change qwen_image_embeds to list for /multi-image
    
    # Qwen tokenization
    # figure out how to shift tokens in this block
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_str}
    ]
    text = lm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    lm_text_tokens = lm_tokenizer([text], return_tensors="pt").to(device)
    lm_text_embeds = lm_model.get_input_embeddings()(lm_text_tokens["input_ids"][0]) # text tokens embedded in a 896-dimensional space
    # CHECK FOR NORMALIZATION OF VECTORS
    lm_embeds = torch.cat((lm_text_embeds[0].unsqueeze_(0), lm_image_embeds[0].unsqueeze(0), lm_text_embeds[1:])) # Unsqueeze to match dimensions

    return lm_embeds

# Integrate to custom model class
def generate(mm_model, lm_model, lm_tokenizer, projector, input_str, input_img):
    mm_embeds = embed_mixed_modal(mm_model.model.model,                                  
                                  mm_model.model.tokenizer,
                                  lm_model,
                                  lm_tokenizer,
                                  projector,
                                  input_str,
                                  input_img)
    
    generated_ids = lm_model.model(inputs_embeds=mm_embeds.unsqueeze_(0))
    print(generated_ids)
    # generated_text = lm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_ids

# Example
print(generate(cpm,
               qwen,
               tokenizer,
               projector,
               "Describe the text I just gave you",
               "/home/jmarie/flares/positive_img/0000.png"
               ))

"""Testing before the generate function was written
multimodal_embeds = embed_mixed_modal(cpm.model.model,
                                      cpm.model.tokenizer,
                                      
generated_ids = qwen.generate(inputs_embeds=multimodal_embeds.unsqueeze_(0), max_new_tokens=512) # literally generated tokens
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
"""
print(dir(qwen))
print("\n\n---------------------------------------------------------\n\n")
print(dir(cpm))

# Intergate to cusom model class
def projector_training_mode(mm_model, lm_model, projector):
    # Freeze multimodal model # Change this to vision tower when it's all better
    mm_model.eval()
    for parameter in mm_model.parameters:
        parameter.requires_grad = False

    # Freeze language model
    lm_model.eval()
    for parameter in lm_model.parameters():
        parameter.requires_grad = False

    projector.train()
    for parameter in projector.parameters():
        parameter.requires_grad = True

    return

# Write documentation
class CLIPLoss(_Loss):
    def __init__():
        super().__init__()
        self.device = "cuda" if torch.cuda.is_visible() else "cpu"

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
      logits = text_embeds @ image_embeds.T
      n = logits.shape[1]
      labels = torch.arange(n)
      logits = logits.to(self.device)

      # Figure out why the lgits.transpose, cf
      # https://github.com/RustamyF/clip-multimodal-ml/blob/main/src/model_loss.py
      # and
      # https://towardsdatascience.com/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72
      images_loss = cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
      texts_loss = cross_entropy(logits, labels, reduction="mean")

      return = (image_loss + text_loss) / 2

# Integrate to custom model class
def train_projector(mm_model, lm_model, projector, data_path):
    """For now, dummy training instance. TO DO: add support for training args
    for more fine-tuned fine-tuning
    TO DO: fuse into model class
    """
    # TEST FOR NORM OF TEXT EMBEDDING
    batch_size = 32
    train, test = load_dataset(dataset)
    loss_fn = # Most important aspect
    projector_training_mode(mm_model, lm_model, projector)

    optimizer = torch.optim.AdamW(projector.parameters()) # Hyperparameters TBP w/ trianing args
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=50) # Same here


    while len(train_dataset) != 0:
        for batch in dataset:
            optimizer.zero_grad()

            output  = lm_model(embed_mixed_modal(mm_model.model, mm_model.tokenizer, input_str, image))
            
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    return

# integrate to custom model class
def training_step(mm_model, lm_model, projector):
    return
