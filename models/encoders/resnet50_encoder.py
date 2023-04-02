import torch
import torch.nn as nn

from models import TorchBaseModel


class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.encoder_raw = torch.hub.load(
            "facebookresearch/dino:main", "dino_resnet50", pretrained=self.pretrained
        )

        self.encoder = nn.Sequential(
            self.encoder_raw.conv1,
            self.encoder_raw.bn1,
            self.encoder_raw.relu,
            self.encoder_raw.maxpool,
            self.encoder_raw.layer1,
            self.encoder_raw.layer2,
            self.encoder_raw.layer3,
        )
        self.encoder_output_size = 1024

    def forward(self, input: torch.Tensor):
        return self.encoder(input)
