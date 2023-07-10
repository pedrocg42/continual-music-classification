import torch
import torch.nn as nn
from transformers import AutoModel


class MertEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__()

        self.pretrained = pretrained

        # Loading model weights
        self.encoder = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )

        self.output_size = 768

    def forward(self, inputs: torch.Tensor):
        outputs = self.encoder(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).permute(
            (1, 0, 2, 3)
        )  # C, B, S, H -> B, C, S, H
        outputs = torch.mean(all_layer_hidden_states, dim=-2)  # B, C, S, H -> B, C, H
        return outputs
