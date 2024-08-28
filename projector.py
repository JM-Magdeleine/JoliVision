"""Definition of CPM to Qwen projector block

TO DO: implement different architectures

Feature ideas:
enable image embedding contextually within text
enable multi-image prompts
"""
import torch
import typing

import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Projector definition
# Integrate to custom model class
class CPMQwenProjector(nn.Module):
    def __init__(self, cpm_dim: int, qwen_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features=cpm_dim, out_features=qwen_dim)
        self.gelu = nn.GELU()
        self.device = device
        self.to(self.device)
        self.to(torch.bfloat16)

        # Default is evaluation mode with random weight
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
