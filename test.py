import inspect
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

# num = 0
# for child in cpm.children():
#     print(num)
#     print(child)
#     num += 1

# print("------------------ NUM --------------------")
# print(num)

"""
modules = []
def add_hook(model):
    def forward_hook_ins(module, input, output):
        print(next(module.named_modules()))
    model.register_forward_hook(forward_hook_ins)
cpm.apply(add_hook)
"""

im_64 = img2base64('/data3/jmarie/internvl-flares-train/001.png')
msgs = [{'role': 'user', 'content': 'What is in the image?'}]
inputs = {"image": im_64, "question": json.dumps(msgs)}

print("chat")
res = cpm.chat(inputs)
print("----------- RES -------------")
print(res)

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
print(dir(qwen))
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
                print("--------------- PARAMETERS ----------------")
                print(parameter)
                if len(parameter.size()) <= 1:
                    nn.init.normal_(parameter)
                else:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, x):
        return self.proj(x)

projector = CPMQwenProjector(input_size=504, output_size=5)
# for layer in projector.modules():
#     print(layer)
# print(sys.path)



def get_image_embeds(input_str: str, image: str) -> torch.Tensor:
    # format input just like model.chat
    # do the same as in generate, but stop when image_bound and input_embeds are there
    # recuperate model_inputs

    # segment out image embeds
    # return cut-out image embeds    

EOT_TOKEN_ID=151644
def tokenize_mixed_modal(input_str: str, image: str) -> List[int]:
    # MiniCPMV tokenization + getting visual tokens
    
    
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
    qwen_tokenized_text_inputs = tokenizer([text], return_tensors="pt").to(device)


tokenize_mixed_modal("[image] Here is the image, [image], show me what this [image] corresponds to", ["/home/jmarie/flares/positive_img/0000.png", "/home/jmarie/flares/positive_img/0001.png", "/home/jmarie/flares/positive_img/0002.png"])
