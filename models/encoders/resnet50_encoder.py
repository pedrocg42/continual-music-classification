import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Encoder(nn.Module):
    def __init__(
        self, pretrained: bool = True, one_channel: bool = True, **kwargs
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.one_channel = one_channel
        self.encoder_raw = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if self.pretrained else None
        )

        self.encoder = nn.Sequential(
            self.encoder_raw.conv1,
            self.encoder_raw.bn1,
            self.encoder_raw.relu,
            self.encoder_raw.maxpool,
            self.encoder_raw.layer1,
            self.encoder_raw.layer2,
            self.encoder_raw.layer3,
            self.encoder_raw.layer4,
        )
        self.encoder_output_size = 2048

        if self.one_channel:
            self.three_to_one_input_channels()

    def forward(self, input: torch.Tensor):
        return self.encoder(input)

    def three_to_one_input_channels(self, dim: int = 1):
        self.encoder[0].weight.data = self.encoder[0].weight.data.sum(
            dim=1, keepdim=True
        )
        self.encoder[0].in_channels = 1
