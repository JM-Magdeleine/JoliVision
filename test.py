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

sys.path.insert(0, "/data3/jmarie/MiniCPM-V")
from chat import MiniCPMVChat, img2base64

cpm = MiniCPMVChat('openbmb/MiniCPM-V-2')

# # num = 0
# # for child in cpm.children():
# #     print(num)
# #     print(child)
# #     num += 1

# # print("------------------ NUM --------------------")
# # print(num)

# """
# modules = []
# def add_hook(model):
#     def forward_hook_ins(module, input, output):
#         print(next(module.named_modules()))
#     model.register_forward_hook(forward_hook_ins)
# cpm.apply(add_hook)
# """

# im_64 = img2base64('/data3/jmarie/internvl-flares-train/001.png')
# msgs = [{'role': 'user', 'content': 'What is in the image?'}]
# inputs = {"image": im_64, "question": json.dumps(msgs)}

# print("chat")
# res = cpm.chat(inputs)
# print("----------- RES -------------")
# print(res)

# # embeds, hidden_states = cpm.model.model.get_vllm_embedding()
# # print("---------- HIDDEN STATES ------------")
# # print(hidden_states)
# # print(cpm.model.tokenizer)
# # print(inspect.getsource(cpm.model.model.chat))
# # print(os.path.abspath(inspect.getfile(cpm.model.model.chat)))
# # print(cpm.model.__dict__)

# # print(cpm.generate_vllm())
# # print("-------------- MODULES ---------------")
# # print(modules)

# # print(cpm.vpm)
# # print(inspect.getmro(type(cpm)))
# # for parameter in cpm.llm.lm_head.parameters():
# #     print(parameter)
# # print(cpm.llm.lm_head.parameters())
# # print(cpm.__dir__())

# # for name, module in cpm.named_modules():
# #     print(name, module)
# # print(cpm.get_vllm_embedding())
# # print(type(cpm).__name__)
# # print(vars(cpm).keys())



from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# print(dir(qwen))
# # Instead of using model.chat(), we directly use model.generate()
# # But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# Directly use generate() and tokenizer.decode() to get the output.
# Use `max_new_tokens` to control the maximum output length.
# generated_ids = qwen.generate(
#     model_inputs.input_ids,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print("-------------- RESPONSE -----------------")
# print(response)

# print("-------------- QWEN -----------------")
class CPMQwenProjector(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=output_size),
            nn.GELU()
        )

        # Xavier init weights
        modules_gen = self.proj.modules()
        next(modules_gen)
        for layer in modules_gen:
            for parameter in layer.parameters():
                # print("--------------- PARAMETERS ----------------")
                # print(parameter)
                if len(parameter.size()) <= 1:
                    nn.init.normal_(parameter)
                else:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, x):
        return self.proj(x)

projector = CPMQwenProjector(input_size=2304, output_size=896)
# for layer in projector.modules():
#     print(layer)
# print(sys.path)


print("----------------- EMBED TEST ------------------")
def get_image_embeds(model, tokenizer, input_str: str, image: str, sampling=True, max_inp_length=2048, **kwargs) -> torch.Tensor:
    vision_hidden_states=None
    
    # format input just like model.chat
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

    generation_config.update(
        (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
    )
        
    # do the same as in generate, but stop when image_bound and input_embeds are loaded
    # MAX_INP_LENGTH ?
    # TOKENIZER ?
    # MAX_NEW_TOKENS ?

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

    # recuperate model_inputs
    image_bound, inputs_embeds = model_inputs["image_bound"], model_inputs["inputs_embeds"]

    # segment out image embeds
    image_embeds = []
    print(image_bound, image_bound[0][0][0])
    image_embeds.append(inputs_embeds[0][image_bound[0][0][0]: image_bound[0][-1][-1]])
    
    # return cut-out image embeds
    return image_embeds

EOT_TOKEN_ID=151644
def embed_mixed_modal(model, cpm_tokenizer, input_str: str, image: str, sampling=True) -> list[int]:
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
