import torch
import torch.nn as nn
from transformers import AutoModel


class MertEncoder(nn.Module):
    def __init__(
        self, pretrained: bool = True, freeze_encoder: bool = True, **kwargs
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        # loading our model weights
        self.encoder = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        
        self.encoder_output_size = 768


    def forward(self, inputs: torch.Tensor):
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        return 
