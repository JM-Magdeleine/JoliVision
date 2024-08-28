"""Definition of CPM Vision tower + projector + Qwen decoder model

WIP: batch processing implementation; LLaVA metrics calculations;
training/testing logger
check for normalization of text embeddings


Ideas: get rid of loading the whole Mini-CPM-V model?
get rid of hardcoded path inesrtion for CPM model
ablation test to check whether encoding vision embeddings in context is useful
    for feature extraction
support for web images
clean up embed image function signature
Shift Qwen inputs to include gap (unk token?) where image embeds will be squeezed in

Feature ideas:
apply support for learning rate customization
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from projector import CPMQwenProjector

import inspect
import traceback

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
# torch.set_default_dtype(torch.bfloat16)


sys.path.insert(0, "/data3/jmarie/MiniCPM-V") # from minicpmv.chat import MiniCPMVChat, img2base64
from chat import MiniCPMVChat, img2base64
cpm = MiniCPMVChat('openbmb/MiniCPM-V-2')


qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", padding_side="left")

IMAGE_PREFIX_TOKENS = torch.tensor(tokenizer("An image containing ").data["input_ids"])
IMAGE_PREFIX_LEN = IMAGE_PREFIX_TOKENS.shape[-1]

class CPMVQwenVLM(nn.Module):
    def __init__(self, mm_model, lm_model, lm_tokenizer, projector):
        super().__init__()
        self.mm_model = mm_model # contains entire CPM model, including tokenizer
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
        and other embeddings necessary for image context. No batch processing yet.
        Heavily borrowed from MiniCPM-V code.
        Args:
            input_str: input string as provided by user
            image: absolute file path to the image to be embedded. Web images are not supported, must be locally saved image
            max_input_length: ????
            kwargs: ??? Used in original MiniCPMV code but idk what use they have here

        Returns:
            `torch.Tensor` 3D tensor:
                The image embeddings, with context included (split token embeddings, ...)
        """
        model = self.mm_model.model.model # extract actual MiniCPMV model
        tokenizer = self.mm_model.model.tokenizer # exctract actual PreTrainedTokenizer tokenizer
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

    def insert_image_embeds(self, lm_image_embeds, lm_text_embeds, lm_text_tokens_list):
        flags = []
        for lm_text_tokens in lm_text_tokens_list:
            flag = 0
            for idx in range(len(lm_text_tokens) - IMAGE_PREFIX_LEN):
                if lm_text_tokens[idx:idx+IMAGE_PREFIX_LEN] == IMAGE_PREFIX_TOKENS:
                    flag = idx + IMAGE_PREFIX_LEN

            if flag == 0:
                # encountered in training
                return [torch.cat((lm_image_embed, lm_text_embed)) for lm_image_embed, lm_text_embed in zip(lm_image_embeds, lm_text_embeds)], [0]*len(lm_text_embeds), [lm_image_embed.shape[-2] for lm_image_embed in lm_image_embeds]
            flags.append(flag)

        return [torch.cat((lm_text_embed[:flag], lm_image_embed, lm_text_embed[flag:]), dim=-1) for flag, lm_text_embed, lm_image_embed in zip(flags, lm_text_embeds, lm_image_embeds)], flags, [lm_image_embed.shape[-2] for lm_image_embed in lm_image_embeds]

    def make_batch(self, tensor_list: list[torch.Tensor], padding_side: str="right", pad_value: int=0):
        """Make N+1 D-tensor corresponding to batch given the list of N D-tensors
        Args:
            tensor_list: list of `torch.Tensor` objects to be made into a batch
            padding_side: string indicating which side to pad to, defaults to left
            pad_value: integer to use for padding, defaults to 0
        
        Returns
            batch: `torch.Tensor` batch tensor of dimensions (`len(tensor_list)`, `tensor.shape`)
            attention_mask: `torch.Tensor` batch tensor of the same dimensions,
                with values of 0 where padding has been applied
        """
        max_len = max([tensor.shape[-2] for tensor in tensor_list])
        batch = torch.full((len(tensor_list), max_len, tensor_list[0].shape[-1]), pad_value)
        attention_mask = torch.zeros_like(batch)

        for i, tensor in enumerate(tensor_list):
            if padding_side == "left":
                batch[i, max_len - tensor.shape[-2]:, :] = tensor
                attention_mask[i, max_len - tensor.shape[-2]:, :] = 1
            else:
                batch[i, :tensor.shape[-2], :] = tensor
                attention_mask[i, :tensor.shape[-2], :] = 1
                
        return batch.to(torch.bfloat16), attention_mask.to(torch.bfloat16)

    # Update doc
    def embed_mixed_modal(self,
                          input_str_list: list[str],
                          image_list: list[str],
                          targets_list: list[str]=None,
                          sampling: bool=True) -> torch.Tensor:
        """Embed mixed-modal data to the Qwen embedding space. The data is embedded
        to the cpm multimodal space. The image embeds are then picked and projected
        into the Qwen embedding space.
        Does: formatting + prompting + tokenization + embedding
        Args:
            input_str_list: lsit of string inputs for processing
            image_list: list of image absolute paths for image inputs. Does not curr support web images
            sampling: idky, it's there + it's used in CPM embedding

        Returns:
            lm_embeds: `torch.Tensor` of shape (batch_size, `len(input_ids)`, `qwen_embedding_space_dims`):
                Embeddings of the multimodal inputs, with the qwen embeds concatenated with
                the complete image embeds
            target_tokens: `torch.Tensor` of shape (batch_size, len(input_ids)):
                Qwen input tokens with -100 where image "tokens" appear, useful for loss calculation by Qwen
        """
        # Maybe shift Qwen inputs prior to embedding? Would help for context for decoder
        
        # Qwen-projected vision embeddings
        mm_image_embeds, _ = self.make_batch([(
            self.embed_image(input_str, image)
        ) for input_str, image in zip(input_str_list, image_list)])
        mm_image_embeds = mm_image_embeds.unsqueeze(0) if mm_image_embeds.dim() == 2 else mm_image_embeds
        image_embeds_len_list = [mm_image_embed.shape[-2] + IMAGE_PREFIX_LEN for mm_image_embed in mm_image_embeds]

        lm_image_embeds = self.projector(mm_image_embeds)
        
        # Qwen tokenization
        if self.projector.training:
            # Captioning task
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


        lm_text_tokens_list = self.lm_tokenizer(text_list)
        lm_text_embeds_list = [self.lm_model.get_input_embeddings()(torch.tensor(lm_text_tokens)) for lm_text_tokens in lm_text_tokens_list["input_ids"]] # text tokens embedded in a 896-dimensional space

        lm_embeds_list, image_start_idx_list, image_embeds_len_list = self.insert_image_embeds(lm_image_embeds, lm_text_embeds_list, lm_text_tokens_list)

        if targets_list is not None:
            targets = self.lm_tokenzier(targets_list)
            targets_list = targets["input_ids"]

            if image_start_idx_list[0] == 0:
                # For size compatibility purposes, check if inserting at the start
                # Insert -100 tokens where image embeds are placed
                masked_targets_list = [
                    torch.cat(
                        (
                            torch.full((1, image_embeds_len), -100),
                            targets_ind.data
                        ),
                        dim=-1
                    ) for image_embeds_len, targets_ind in zip(image_embeds_len_list, targets_list)
                ]
            else:
                masked_targets_list = [
                    torch.cat(
                        (
                            targets_ind.data[:image_start_idx],
                            torch.full((1, image_embeds_len), -100),
                            targets_ind.data[image_start_idx:]
                        ),
                        dim=-1
                    ) for image_start_idx, image_embeds_len, targets_ind in zip(image_start_idx_list, image_embeds_len_list, targets_list)
                ]
            masked_targets, attention_mask = self.make_batch(tensor_list=masked_targets_list, pad_value=0)
            targets.data["input_ids"] = masked_targets
            targets.data["attention_mask"] = attention_mask

        else:
            targets = None

        lm_embeds, attentions = self.make_batch(lm_embeds_list, pad_value=0)

        if targets is not None:
            return lm_embeds, targets
        else:
            return lm_embeds, attentions

    def generate(self, text_list: list[str], image_list: list[str]):
        """Can only get it to work with batch_size 1 for the moment,
        does not want to recognize padding for some reason, so it will
        raise a ValueError stating that padding_side="right" with the tokenizer
        """
        mm_embeds, _ = self.embed_mixed_modal(text_list, image_list)
        mm_embeds = mm_embeds.squeeze(0)
        generated_ids = self.lm_model.generate(
            inputs_embeds=mm_embeds.unsqueeze(0),
            max_length=2048
        )
        generated_text = self.lm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

    def forward(self, text_list, image_list, targets_list=None):
        mm_embeds, targets = self.embed_mixed_modal(text_list, image_list, targets_list)
        res = self.lm_model(
            inputs_embeds=mm_embeds.unsqueeze(0),
            labels=targets["input_ids"],
            attention_mask=targets["attention_mask"]
        )

        return res

    def projector_eval_mode(self):
        # For evaluating/generating
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
        # For training only the projector
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
        """Main training function
        """
        log_frequency = 10 # How many training steps to be done between two train loss logs
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
        train_step, eval_step, train_loss = 0, 0, torch.zeros(batch_size)
        
        # Split the dataset into batches, data points represented by their indices -> to go thru sequentially
        indices = random.shuffle([i for i in ragen(len(train))])
        last_batch_size = len(train) % batch_size
        
        batches = [indices[curr_idx:curr_idx+batch_size] for curr_idx in range(len(train)//batch_size)]
        if last_batch_size == 1:
            batch_list += [indices[-1]]
        else:
            batch_list += [idx for idx in indices[len(train)-last_batch_len:]] # review that line
        
        # Training loop
        for curr_epoch in range(epochs):
          for training_step, curr_batch in tqdm(enumerate(batch_list), desc=f"Current epoch: {curr_epoch+1}"):
              try:
                  caption_list = [train[idx]["conversations"][1]["value"] for idx in curr_batch]
                  image = [os.path.join(data_path, train[idx]["image"]) for idx in curr_batch]

                  result = self.forward(image_list=image_lsit, text_list=caption_list, targets_list=caption_list)
                  loss = result["loss"]
                  train_loss += torch.sum(loss)/batch_size

                  if training_step % 1000 == 0:
                      self.projector.save(save_path)
                      print(train_loss)
                      # self.eval_projector(load_path=save_path, dataset=train, eval_step=eval_step)

                  if training_step % 10 == 0:
                      # Revisit the logging
                      log.append({"epoch": curr_epcoh, "training_step": training_step, "loss": train_loss/10})
                      train_loss = torch.zeros(batch_size)

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

projector = CPMQwenProjector(cpm_dim=2304, qwen_dim=896)
vlm = CPMVQwenVLM(cpm, qwen, tokenizer, projector)

# Example
print(vlm.generate(
      ["Describe the image I just gave you"],
      ["/home/jmarie/flares/positive_img/0000.png"]
    )
)


# vlm.train_projector(save_path="/data3/jmarie/JoliVision/test-checkpoint/test.pt", lr=1e-2, testing=False)

vlm.eval_projector("/data3/jmarie/JoliVision/test-checkpoint/test.pt", save_results=True)

# vlm_random = CPMVQwenVLM(cpm, qwen, tokenizer, CPMVQwenProjector(cpm_dim=2304, qwen_dim=896).bfloat16())
# vlm_random.eval_projector()
