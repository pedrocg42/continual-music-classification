import torch
import torch.nn as nn
from music_genre_classification.models.encoders import EncoderFactory
from music_genre_classification.models.bottlenecks import BottleneckFactory


class TorchClassificationModel(nn.Module):
    def __init__(
        self,
        encoder: dict,
        num_classes: int,
        pooling_type: str = "mert_mean",
        dropout: float = 0.3,
        head_config: dict = [
            {
                "layer_type": "conv1d",
                "in_channels": 13,
                "out_channels": 1,
                "kernel_size": 1,
            },
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
        frozen_encoder: bool = False,
        frozen_decoder: bool = False,
        **kwargs
    ):
        super().__init__()
        self.encoder = EncoderFactory.build(encoder)
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.head_config = head_config
        self.bottleneck = BottleneckFactory.build(bottleneck)
        self.frozen_encoder = frozen_encoder
        self.frozen_decoder = frozen_decoder

        self.initialize()

    def initialize_encoder(self):
        if self.pooling_type == "mean":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pooling_type == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        elif self.pooling_type == "mert_mean":
            self.pooling = nn.AdaptiveAvgPool2d((self.encoder.encoder_output_size))

        if self.frozen_encoder:
            self.freeze_encoder()

    def initialize_decoder(self):
        decoder = []
        in_features = self.encoder.encoder_output_size
        for layer_dict in self.head_config:
            if layer_dict["layer_type"] == "linear":
                decoder.append(
                    nn.Linear(
                        in_features=in_features, out_features=layer_dict["out_features"]
                    )
                )
                in_features = layer_dict["out_features"]
            elif layer_dict["layer_type"] == "conv1d":
                decoder.append(
                    nn.Conv1d(
                        in_channels=layer_dict["in_channels"],
                        out_channels=layer_dict["out_channels"],
                        kernel_size=layer_dict["kernel_size"],
                    )
                )
                decoder.append(nn.Flatten())
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

        if self.frozen_decoder:
            self.freeze_decoder()

    def initialize(self):
        self.initialize_encoder()
        self.initialize_decoder()

    def forward(self, inputs: torch.Tensor):
        with torch.no_grad():
            outputs = self.encoder(inputs)
        if self.bottleneck is not None:
            outputs = self.bottleneck(outputs)
        # outputs = self.pooling(outputs)
        outputs = self.decoder(outputs)
        return outputs

    def prepare_train(self):
        if self.frozen_encoder:
            self.encoder.eval()
        else:
            self.encoder.train()

        if self.bottleneck is not None:
            self.bottleneck.eval()

        if self.frozen_decoder:
            self.decoder.eval()
        else:
            self.decoder.train()

    def prepare_eval(self):
        self.encoder.eval()
        if self.bottleneck is not None:
            self.bottleneck.eval()
        self.decoder.eval()

    def prepare_keys_initialization(self):
        self.encoder.eval()
        self.bottleneck.train()
        self.decoder.eval()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.frozen_encoder = True

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        self.frozen_decoder = True
