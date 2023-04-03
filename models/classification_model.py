import torch
import torch.nn as nn

from models.torch_base_model import TorchBaseModel
from models.encoders import ResNet50Encoder


class TorchClassificationModel(TorchBaseModel):
    def __init__(
        self,
        encoder_name: str,
        pretrained: bool,
        num_classes: int,
        pooling_type: str = "max",
        dropout: float = 0.3,
        head_config: dict = [
            {"layer_type": "linear", "out_features": 256},
            {"layer_type": "bn", "num_features": 256},
            {"layer_type": "relu"},
            {"layer_type": "dropout"},
            {"layer_type": "linear", "out_features": 128},
            {"layer_type": "bn", "num_features": 128},
            {"layer_type": "relu"},
            {"layer_type": "dropout"},
        ],
        **kwargs
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.head_config = head_config

        self.initialize()

    def initialize(self):
        if self.encoder_name == "dino_resnet50":
            encoder_class = ResNet50Encoder
        else:
            raise NotImplementedError("Please select an implemented backbone")

        self.encoder = encoder_class(pretrained=self.pretrained)

        head = []
        if self.pooling_type == "avg":
            head.append(nn.AdaptiveAvgPool2d((1, 1)))
            head.append(nn.Flatten())
        elif self.pooling_type == "max":
            head.append(nn.AdaptiveMaxPool2d((1, 1)))
            head.append(nn.Flatten())

        if self.dropout is not None:
            head.append(nn.Dropout(p=self.dropout))

        for i, layer_dict in enumerate(self.head_config):
            if i == 0:
                in_features = self.encoder.encoder_output_size

            if layer_dict["layer_type"] == "linear":
                head.append(
                    nn.Linear(
                        in_features=in_features, out_features=layer_dict["out_features"]
                    )
                )
                in_features = layer_dict["out_features"]
            elif layer_dict["layer_type"] == "bn":
                head.append(nn.BatchNorm1d(num_features=layer_dict["num_features"]))
            elif layer_dict["layer_type"] == "relu":
                head.append(nn.ReLU())
            elif layer_dict["layer_type"] == "relu6":
                head.append(nn.ReLU6())
            elif layer_dict["layer_type"] == "dropout":
                head.append(nn.Dropout1d(p=layer_dict.get("p", self.dropout)))
        head.append(nn.Linear(in_features=in_features, out_features=self.num_classes))
        self.head = nn.Sequential(*head)

        self.model = nn.Sequential(self.encoder, self.head)

    def inference(self, inputs: torch.Tensor, **kwargs):
        return self.model(inputs)
