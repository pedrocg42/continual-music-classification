import torch
import torch.nn as nn
from clmr.models.sample_cnn_xl import SampleCNNXL


class ClmrEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__()

        self.pretrained = pretrained

        # Loading model weights
        self.encoder = SampleCNNXL(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.output_size = 2048

    def forward(self, inputs: torch.Tensor):
        outputs = self.encoder(**inputs, output_hidden_states=True)
        return outputs
