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
import os
import sys
import torch
import typing

from tqdm import tqdm
from torch import nn
from PIL import Image
from torch.nn.functional import cross_entropy
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Testing imports
import inspect

# TO DO need to remove this line to get rid of local dependencies,
# but needed for CPMV classes ˇˇˇˇˇˇˇˇˇˇ
sys.path.insert(0, "/data3/jmarie/MiniCPM-V")
from chat import MiniCPMVChat, img2base64

cpm = MiniCPMVChat('openbmb/MiniCPM-V-2')
device = "cuda" if torch.cuda.is_available() else "cpu"

qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

def image_bound_to_mask(embeds, image_bound):
    image_seq_len = [ind_image_bound[-1][-1]-ind_image_bound[0][0] for ind_image_bound in image_bound]
    mask = torch.ones(embeds.shape[:-1])
    for idx, element in enumerate(embeds):
        mask[idx, 1:(image_seq_len[idx]+1)] = 0
    
    return mask


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

projector = CPMQwenProjector(cpm_dim=2304, qwen_dim=896).to(device).bfloat16() # TO DO moving to device and changing data type incorporated into class __init__ for completing /projector-class # fixfixfixfixfixfixfix

dataset = load_dataset("liuhaotian/LLaVA-Pretrain")
print(dataset)

class CPMVQwenVLM(nn.Module):
    # CLEAN UP THE DAMN CLASS, mm_model, mm_model.model.model, mm_model.model.tokenizer ????
    def __init__(self, mm_model, lm_model, lm_tokenizer, projector):
        super().__init__()
        self.mm_model = mm_model
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer
        self.projector = projector

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def embed_image(self,
                    input_str: str,
                    image: str,
                    max_inp_length: int=2048,
                    **kwargs) -> torch.Tensor:
        """Get image embeddings from MiniCPMV's vision tower, including slice embeddings
        and other embeddings necessary for image context. (Get a list of em?)
        Args:
            input_str: input string as provided by user
            image: ABSOLUTE file path to the image to be embedded. Web images are not supported, must be locally saved image
            max_input_length: ????
            kwargs: ??? Used in original MiniCPMV code but idk what use they have here

        Returns:
            `torch.Tensor` of shape (1, `len(image_embeds)`, `cpm_embedding_dim`):
                The image embeddings, with context included (split token embeddings, ...)
        """
        # Explore whether the whole input conditioning is better for projector training
        # Rethink using embeddings within the context of the sentence
        # (can it learn the influence of the position of the image in the sentence?)

        model = self.mm_model.model.model # class MiniCPMV
        tokenizer = self.mm_model.model.tokenizer # class PreTrainedTokenizer
        vision_hidden_states = None # Necessary?
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

        # Dig into MiniCPM source code, find out how the hidden states are integrated
        # into the input embeds from the vision embedding (get_vision_embedding),
        # then get rid of overkill get_vllm_embedding call
        with torch.no_grad():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = model.get_vllm_embedding(model_inputs)

        # Recuperate embedded model_inputs, segment out image embeds
        image_bound, inputs_embeds = model_inputs["image_bound"], model_inputs["inputs_embeds"]
        image_embeds = inputs_embeds[0][image_bound[0][0][0]: image_bound[0][-1][-1]] # for multiple image inference, replace image_embeds with a list to append to + change image_bound indices

        return image_embeds

    # Update doc
    def embed_mixed_modal(self,
                          input_str: str,
                          image: str,
                          target: str=None,
                          sampling: bool=True) -> list[int]:
        """Embed mixed-modal data to the Qwen embedding space. The data is embedded
        to the cpm multimodal space. The image embeds are picked out and subsequently
        projected to the Qwen embedding space using a linear GELU unit.
        All inputs preparation for embedding is done here (formatting + prompting + tokenization)
        Args:
            input_str: the complete string input from the user
            image: image absolute path as image input. Does NOT curr support web images
            sampling: idky, it's there, and it's used in CPM embedding

        Returns:
            lm_embeds: `torch.Tensor` of shape (1 (/batch_len), `len(input_ids)`, `qwen_embedding_space_dims`):
                Embeddings of the multimodal inputs, with the picture projected embeddings
                before the qwen text embeds
            image_mask: `torch.Tensor` of shape (1, len(input_ids)), 0 indicating the presence of an image or image-related token
        """
        # Maybe shift Qwen inputs prior to embedding? Would help for context for decoder
        # MiniCPMV tokenization + getting visual embeddings

        mm_image_embeds = self.embed_image(input_str, image)

        lm_image_embeds = self.projector(mm_image_embeds) # change lm_image_embeds to list.append for /multi-image

        # Qwen tokenization
        # figure out how to shift tokens in this block
        if self.projector.training:
            text = input_str

        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_str}
            ]
            text = self.lm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        lm_text_tokens = self.lm_tokenizer([text], return_tensors="pt").to(device)
        target_tok = self.lm_tokenizer([text], return_tensors="pt").to(device)
        lm_text_embeds = self.lm_model.get_input_embeddings()(lm_text_tokens["input_ids"][0]) # text tokens embedded in a 896-dimensional space
        print("LM TEXT EMBEDS", lm_text_embeds.shape)
        # CHECK FOR NORMALIZATION OF VECTORS
        lm_embeds = torch.cat((lm_text_embeds[0].unsqueeze_(0), lm_image_embeds, lm_text_embeds[1:])) # Unsqueeze to match dimensions
        if target is not None:
            # image_mask = image_bound_to_mask(lm_embeds, image_bound) if target is not None else None
            masked_target = torch.cat((target_tokens[0].unsqueeze(0), torch.fill((lm_image_embeds.shape[0]), target_tokens[1:]), -100))
        else:
            masked_target = None

        return lm_embeds, masked_target

    def generate(self, text: str, image: str):
        mm_embeds, _ = self.embed_mixed_modal(text, image)

        generated_ids = self.lm_model.generate(inputs_embeds=mm_embeds.unsqueeze_(0), max_length=2048)
        generated_text = self.lm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

    def forward(self, text, image, target=None):
        mm_embeds, masked_labels = self.embed_mixed_modal(text, image, target)
        res = self.lm_model(inputs_embeds=mm_embeds[...,:-1, :].unsqueeze(0), labels=masked_labels)

        return res

    def projector_training_mode(self):
        # Freeze multimodal model
        self.mm_model.model.model.eval()
        for parameter in self.mm_model.model.model.parameters():
            parameter.requires_grad = False

        # Freeze language model
        self.lm_model.eval()
        for parameter in self.lm_model.parameters():
            parameter.requires_grad = False

        self.projector.train()
        for parameter in self.projector.parameters():
            parameter.requires_grad = True

        return

    def train_projector(
            self,
            save_path,
            batch_size: int=1,
            lr: float=1e-3,
            dataset=None,
        ):

        """For now, dummy training instance, using CLIP dataset
        implement dataloader, add dataset_path argument to fn signature
        TO DO: allow for passing of training args for more fine-tuned fine-tuning
        """
        # TEST FOR NORM OF TEXT EMBEDDING which dimensions ?
        if dataset is not None:
            train, test = load_dataset(dataset)

        else:
            train, test = None, None

        self.projector_training_mode()

        optimizer = torch.optim.AdamW(self.projector.parameters()) # Hyperparameters TBP w/ trianing args
        # lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=50) # Same here, also implement WSD act MiniCPM


        # Start loop for batch processing ----> ???
        optimizer.zero_grad()
        if dataset is None:
            # Default dataset is LLaVA Instruct 595k augmented with SNCF dataset
            data_path = "/data3/jmarie/JoliVision/LLaVA-CC3M-Pretrain-595k/images/"
            with open("/data3/jmarie/JoliVision/LLaVA-CC3M-Pretrain-595k/chat-aug.json") as dataset_reader:
                dataset = json.load(file_reader)
        for data_point in tqdm(dataset):
            try:
                conversation = data_point["prompt"]
                image = os.path.join(data_path, data_point["image"])
                result = self.forward(image=image, text=description, target=description)
                loss = result["loss"]
                print(loss)
                
                loss.backward()
                optimizer.step()
          
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(exc_traceback)
                print(f"The following error was raison: {exc_type}")
                print(f"Training was terminated for the follwing reason: {exc_value}")
                
                self.projector.save(save_path) # Change to save the state with the training progress as well
                
                sys.exit(1)

        self.projector.save(save_path)

        return

    def test_trained_projector(self, load_path):
        self.projector.load_projector_checkpoint(load_path)

        with open("/data3/jmarie/JoliVision/test-dataset.json") as file_reader:
            dataset = json.load(file_reader)

        for data_point in tqdm(dataset):
            description = data_point["prompt"]
            image = data_point["image"]
            print(image.split("/")[-1], self.generate("Repeat the past text.", image))
        
        return


# Example
"""print(generate(
      "Describe the image I just gave you",
      "/home/jmarie/flares/positive_img/0000.png"
    )
)
"""

# print(cpm.model.model.__dict__.keys())
# print(inspect.getmro(type(cpm.model.model)))
vlm = CPMVQwenVLM(cpm, qwen, tokenizer, projector)
print(os.path.abspath(inspect.getfile(qwen.forward)))
#vlm.train_projector(save_path="/data3/jmarie/JoliVision/test-checkpoint/test.pt", dataset="liuhaotian/LLaVA-Pretrain")

#vlm.test_trained_projector("/data3/jmarie/JoliVision/test-checkpoint/test.pt")
