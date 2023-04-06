import torch
import torch.nn as nn

from models.torch_base_model import TorchBaseModel


class TorchClassificationModel(TorchBaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        pooling_type: str = "max",
        dropout: float = 0.3,
        head_config: dict = [
            {"layer_type": "dropout"},
            {"layer_type": "linear", "out_features": 256},
            {"layer_type": "bn", "num_features": 256},
            {"layer_type": "relu"},
            {"layer_type": "dropout"},
            {"layer_type": "linear", "out_features": 128},
            {"layer_type": "bn", "num_features": 128},
            {"layer_type": "relu"},
            {"layer_type": "dropout"},
        ],
        bottleneck: nn.Module = None,
        **kwargs
    ):
        super().__init__()
        self.encoder_raw = encoder
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.head_config = head_config
        self.bottleneck = bottleneck

        self.initialize()

    def initialize_encoder(self):
        pooling = []
        if self.pooling_type == "avg":
            pooling.append(nn.AdaptiveAvgPool2d((1, 1)))
            pooling.append(nn.Flatten())
        elif self.pooling_type == "max":
            pooling.append(nn.AdaptiveMaxPool2d((1, 1)))
            pooling.append(nn.Flatten())

        self.encoder = nn.Sequential(self.encoder_raw, *pooling)

    def initialize_decoder(self):
        decoder = []
        for i, layer_dict in enumerate(self.head_config):
            if i == 0:
                in_features = self.encoder_raw.encoder_output_size

            if layer_dict["layer_type"] == "linear":
                decoder.append(
                    nn.Linear(
                        in_features=in_features, out_features=layer_dict["out_features"]
                    )
                )
                in_features = layer_dict["out_features"]
            elif layer_dict["layer_type"] == "bn":
                decoder.append(nn.BatchNorm1d(num_features=layer_dict["num_features"]))
            elif layer_dict["layer_type"] == "relu":
                decoder.append(nn.ReLU())
            elif layer_dict["layer_type"] == "relu6":
                decoder.append(nn.ReLU6())
            elif layer_dict["layer_type"] == "dropout":
                decoder.append(nn.Dropout1d(p=layer_dict.get("p", self.dropout)))
        decoder.append(
            nn.Linear(in_features=in_features, out_features=self.num_classes)
        )

        self.decoder = nn.Sequential(*decoder)

    def initialize(self):
        self.initialize_encoder()
        self.initialize_decoder()

    def inference(self, inputs: torch.Tensor):
        output = self.encoder(inputs)
        if self.bottleneck is not None:
            output = self.bottleneck(output)
        output = self.decoder(output)
        return output
