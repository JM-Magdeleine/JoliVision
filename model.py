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

from projector import CPMQwenProjector

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
        image_embeds = inputs_embeds[0][image_bound[0][0][0]: image_bound[0][-1][-1]]

        return image_embeds

    def insert_image_embeds(self, lm_image_embeds, lm_text_embeds, lm_text_tokens):
        flags = []
        for seq_idx in range(lm_text_tokens.shape[0]):
            flag = 0
            for idx in range(len(lm_text_tokens) - IMAGE_PREFIX_LEN):
                if lm_text_tokens[idx:idx+IMAGE_PREFIX_LEN] == IMAGE_PREFIX_TOKENS:
                    flag = idx + IMAGE_PREFIX_LEN

            if flag == 0:
                return torch.cat((lm_image_embeds, lm_text_embeds)), 0, lm_image_embeds.shape[-2]
            flags.append(flag)

        return [torch.cat((lm_text_embed[:flag], lm_image_embed, lm_text_embed[flag:]), dim=-1) for flag, lm_text_embed, lm_image_embed in zip(flag, lm_text_embeds, lm_image_embeds)], flags, [lm_image_embed.shape[-2] for lm_image_embed in lm_image_embeds]

    def pad_batch(tensors_list: list[torch.Tensor],
                  padding_side: str=None
                  ) -> torch.Tensor:
        max_size = max([tensor.shape[0] for tensor in tensors_list])
        pads = [torch.tensor(1+max_size-tensor.shape[0]) for tensor in tensors_list]

        if padding_side is None:
            # Padding is right by default
            batch = torch.stack([torch.cat((pad, tensor)) for pad, tensor in zip(pads, tensors_list) if ])
        
        return batch[:, 1:, ...]

    # Update doc
    def embed_mixed_modal(self,
                          input_str_list: list[str],
                          image_list: list[str],
                          target_list: list[str]=None,
                          sampling: bool=True) -> torch.Tensor:
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

        mm_image_embeds = torch.cat(
            [self.embed_image(input_str, image) for input_str, image in zip(input_str_list, image_list)],
            dim=0
        )
        image_embeds_len_list = [mm_image_embed.shape[-2] + IMAGE_PREFIX_LEN for mm_image_embed in mm_image_embeds]

        lm_image_embeds = self.projector(mm_image_embeds)

        # Qwen tokenization
        # figure out how to shift tokens in this block
        if self.projector.training:
            text_list = ["An image containing " + input_str for input_str in input_str_list]

        else:
            messages_list = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "An image containing " + input_str}
                ] for input_str in input_str_list
            ]
            text_list = self.lm_tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True
            )


        lm_text_tokens_list = self.lm_tokenizer(text_list, return_tensors="pt")
        lm_text_embeds = self.lm_model.get_input_embeddings()(lm_text_tokens["input_ids"]) # text tokens embedded in a 896-dimensional space

        lm_embeds_list, image_start_idx_list, image_embeds_len_list = self.insert_image_embeds(lm_image_embeds, lm_text_embeds, lm_text_tokens_list)

        # CHECK FOR NORMALIZATION OF VECTORS
        if target_list is not None:
            target_tokens_list = self.lm_tokenizer(text_list, return_tensors="pt")

            if image_start_idx_list[0] == 0:
                masked_targets_list = [
                    torch.cat(
                        (
                            torch.full((1, image_embeds_len), -100),
                            target_tokens.data["input_ids"]
                        ),
                        dim=-1
                    ) for image_embeds_len in image_embeds_len_list
                ]
            else:
                masked_targets_list = [
                    torch.cat(
                        (
                            target_tokens.data["input_ids"][:image_start_idx],
                            torch.full((1, image_embeds_len), -100),
                            target_tokens.data["input_ids"][image_start_idx:]
                        ),
                        dim=-1
                    ) for image_start_idx, image_embeds_len in zip(image_start_idx_list, image_embeds_len_list)
                ]
            masked_targets = pad_batch(masked_targets_list)
            target_tokens.data["input_ids"] = masked_targets
            target_tokens.data["attention_mask"] = torch.ones_like(masked_targets)

        else:
            target_tokens = None

        lm_embeds = pad_batch(lm_embeds_list)

        return lm_embeds, target_tokens

    def make_batch(self, ):

      return

    def generate(self, text: str, image: str):
        batch_embeds, _ = make_batch(...)
        mm_embeds, _ = self.embed_mixed_modal(text, image)

        generated_ids = self.lm_model.generate(
            inputs_embeds=mm_embeds.unsqueeze_(0),
            max_length=2048
        )
        generated_text = self.lm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

    def forward(self, text, image, target=None):
        mm_embeds, labels_with_mask = self.embed_mixed_modal(text, image, target)
        res = self.lm_model(
            inputs_embeds=mm_embeds.unsqueeze(0),
            labels=labels_with_mask
        )

        return res

    def projector_eval_mode(self):
        self.mm_model.model.model.eval()
        for parameter in self.mm_model.model.model.parameters():
            parameter.requires_grad = False

        self.lm_model.eval()
        for parameter in self.lm_model.parameters():
            parameter.requires_grad = False

        self.projector.eval()
        for parameter in self.projector.parameters():
            parameter.requires_grad = False

        return

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
            train=None,
            test=None,
            testing=False
        ):
        log_frequency = 10 # How many training steps to be done between two train loss logs
        # TEST FOR NORM OF TEXT EMBEDDING which dimensions ?
        if dataset is not None:
            train, test = load_dataset(dataset)

        if dataset is None and train is None and test is None:
            # Default dataset is LLaVA Instruct 595k augmented with SNCF dataset
            data_path = "/data3/jmarie/JoliVision/LLaVA-CC3M-Pretrain-595K/images/"
            with open("/data3/jmarie/JoliVision/LLaVA-CC3M-Pretrain-595K/mini-llava-train.json") as train_dataset_reader:
                train = json.load(train_dataset_reader)
            with open("/data3/jmarie/JoliVision/LLaVA-CC3M-Pretrain-595K/mini-llava-test.json") as test_dataset_reader:
                test = json.load(test_dataset_reader)


        self.projector_training_mode()

        optimizer = torch.optim.AdamW(self.projector.parameters(), lr=lr) # Hyperparameters TBP w/ trianing args
        # lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=50) # Same here, also implement WSD act MiniCPM

        optimizer.zero_grad()

        log = []
        train_step, eval_step, train_loss = 0, 0, torch.zeros(1)
        for data_point in tqdm(train):
            try:
                caption = data_point["conversations"][1]["value"]
                image = os.path.join(data_path, data_point["image"])

                result = self.forward(image=image, text=caption, target=caption)
                loss = result["loss"]
                train_loss += loss

                if train_step % 10 == 0:
                    # Revisit the logging
                    log.append(f"Training step {train_step} loss: {(train_loss/10).tolist()[0]}")
                    train_loss = torch.zeros(1)

                if train_step % 1000 == 0:
                    self.projector.save(save_path)
                    print(train_loss)
                    # self.eval_projector(load_path=save_path, dataset=train, eval_step=eval_step)
                    # eval_step += 1

                loss.backward()
                optimizer.step()

            except:
                traceback.print_exc()
                self.projector.save(save_path) # Change to save the state with the training progress as well
                print("projector saved!")

                with open("/data3/jmarie/JoliVision/test-checkpoint/training-log.txt", "w") as train_logger:
                    json.dump(log, train_logger)
                    print("training log successfully saved")

                sys.exit(1)

            train_step += 1

        self.projector.save(save_path)

        # Revisit the logging
        with open("/data3/jmarie/JoliVision/test-checkpoint/training-log.txt", "w") as train_logger:
            json.dump(log, train_logger)

        if testing:
            self.eval_projector(load_path=save_path, dataset=test)

        self.projector_eval_mode()

        return

    def eval_projector(
        self,
        load_path: str=None,
        dataset=None,
        eval_step: int=-1,
        save_results: bool=False,
        result_save_file: str=None,
        log_details: str=None
        ):
        
        # In development
        if load_path is not None:
            self.projector.load_projector_checkpoint(load_path)
        self.projector_eval_mode()

        data_path = "/data3/jmarie/JoliVision/LLaVA-CC3M-Pretrain-595K/images/"

        if dataset is None:
            with open("/data3/jmarie/JoliVision/LLaVA-CC3M-Pretrain-595K/mini-llava-test.json") as file_reader:
                dataset = json.load(file_reader)

        print(torch.cuda.mem_get_info())

        total_loss = torch.zeros(1)
        for data_point in tqdm(dataset):
            caption = data_point["conversations"][1]["value"]
            image = os.path.join(data_path, data_point["image"])
            result = self.forward(image=image, text=caption, target=caption)

            total_loss += result["loss"]

        if save_results and result_save_file is None:
            with open("/data3/jmarie/JoliVision/training_log.txt") as test_logger:
                train_logger.write(f"Eval step {eval_step} testing loss: {total_loss[0]/len(dataset)}\n")
                
                if log_details is not None:
                  train_logger.write(f"{log_details}\n")

            return

def make_lm_batch(text_embeds: list[torch.Tensor],
                  image_embeds: list[torch.Tensor],
                  image_flags: list[int],
                  image_embeds_len_list: list[int],
                  padding_side: str=None
                  ):
    max_size = max([text_embed.shape[-1] + image_embed.shape[-1] for text_embed, image_embed in zip(text_embeds, image_embeds)])
    max_size_idx = [i for i in range(len(text_embeds)) if text_embeds[i].shape[-1]+image_embeds.shape[-1] == max]
    
    pad_sizes = [torch.zeros(text_embed.shape[0], max_size-text_embed.shape[-1]-image_embed.shape[-1]) for idx, text_embed, image_embed in enumerate(zip(text_embeds, image_embeds)) if idx not in max_size_idx else torch.empty(image_embeds[0].shape[-1], 1)]

    batch = []
    for i, text_embed, image_embed in enumerate(zip(text_embeds, image_embeds)):
        batch.append(torch.cat((padding_list[i], text_embed[:image_flags[i]], image_embed, text_embed[image_flags[i]+image_embeds_len_list[i]]), dim=-1))

    if padding_side is None:
        for idx in range()
        # Default padding is right
        return torch.stack([torch.cat((padding, text_embed[:image_flag], image_embed, text_embed[image_flag:, :]), dim=-1) for i, padding, text_embed[:, :image_flags]], dim=0)


projector = CPMQwenProjector(cpm_dim=2304, qwen_dim=896).bfloat16()
# print(cpm.model.model.__dict__.keys())
# print(inspect.getmro(type(cpm.model.model)))
vlm = CPMVQwenVLM(cpm, qwen, tokenizer, projector)
# print(os.path.abspath(inspect.getfile(qwen.forward)))

# Example

print(vlm.generate(
      "Describe the image I just gave you",
      "/home/jmarie/flares/positive_img/0000.png"
    )
)


# vlm.train_projector(save_path="/data3/jmarie/JoliVision/test-checkpoint/test.pt", lr=1e-2, testing=False)

vlm.eval_projector("/data3/jmarie/JoliVision/test-checkpoint/test.pt", save_results=True)

# vlm_random = CPMVQwenVLM(cpm, qwen, tokenizer, CPMVQwenProjector(cpm_dim=2304, qwen_dim=896).bfloat16())
# vlm_random.eval_projector()

def stack_tensors_with_padding(tensors, padding_value=0, dim=0):
    # Determine the maximum size in each dimension
    max_sizes = [max(sizes) for sizes in zip(*[tensor.shape for tensor in tensors])]
    
    # Pad each tensor to match the maximum size
    padded_tensors = []
    for tensor in tensors:
        pad_sizes = []
        for i, size in enumerate(tensor.shape):
            pad_before = 0
            pad_after = max_sizes[i] - size
            pad_sizes.append((pad_before, pad_after))
        
        # Flatten pad_sizes to match the required input format for torch.nn.functional.pad
        pad_sizes = [p for pair in reversed(pad_sizes) for p in pair]
        
        # Pad the tensor
        padded_tensor = torch.nn.functional.pad(tensor, pad_sizes, value=padding_value)
        padded_tensors.append(padded_tensor)
    
    # Stack the padded tensors along the specified dimension
    stacked_tensor = torch.stack(padded_tensors, dim=dim)
    
    return stacked_tensor

# Example usage:
tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(3, 2)
tensor3 = torch.randn(1, 4)

stacked_tensor = stack_tensors_with_padding([tensor1, tensor2, tensor3])
print(stacked_tensor)
