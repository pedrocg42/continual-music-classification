import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Encoder(nn.Module):
    def __init__(
        self, pretrained: bool = True, one_channel: bool = True, **kwargs
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.one_channel = one_channel
        encoder_raw = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if self.pretrained else None
        )

        self.encoder = nn.Sequential(
            encoder_raw.conv1,
            encoder_raw.bn1,
            encoder_raw.relu,
            encoder_raw.maxpool,
            encoder_raw.layer1,
            encoder_raw.layer2,
            encoder_raw.layer3,
            encoder_raw.layer4,
        )
        self.output_size = 2048

        if self.one_channel:
            self.three_to_one_input_channels()

    def forward(self, input: torch.Tensor):
        return self.encoder(input)

    def three_to_one_input_channels(self, dim: int = 1):
        self.encoder[0].weight.data = self.encoder[0].weight.data.sum(
            dim=1, keepdim=True
        )
        self.encoder[0].in_channels = 1
