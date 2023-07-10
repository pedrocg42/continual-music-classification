import copy

import torch
import torch.nn as nn

import config
from music_genre_classification.models.bottlenecks import BottleneckFactory
from music_genre_classification.models.encoders import EncoderFactory


class MertClassificationDecoder(nn.Module):
    def __init__(
        self,
        conv1_dict: dict[str, int],
        in_features: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=conv1_dict["in_channels"],
            out_channels=conv1_dict["out_channels"],
            kernel_size=conv1_dict["kernel_size"],
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1d(inputs)
        outputs = self.flatten(outputs)
        outputs = self.fc(outputs)
        return outputs


class TorchClassificationModel(nn.Module):
    def __init__(
        self,
        encoder: dict,
        num_classes: int,
        pooling_type: str | None = None,
        dropout: float = 0.3,
        head_config: list[dict] = [
            {
                "layer_type": "conv1d",
                "in_channels": 13,
                "out_channels": 1,
                "kernel_size": 1,
            },
        ],
        bottleneck: nn.Module | None = None,
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
        else:
            self.pooling = None

        if self.frozen_encoder:
            self.freeze_encoder()

    def initialize_decoder(self):
        decoder = []
        in_features = self.encoder.output_size
        for layer_dict in self.head_config:
            if layer_dict["layer_type"] == "linear":
                decoder.append(
                    nn.Linear(
                        in_features=in_features,
                        out_features=layer_dict["out_features"]
                        if "out_features" in layer_dict
                        else self.num_classes,
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
            elif layer_dict["layer_type"] == "flatten":
                decoder.append(nn.Flatten())
            elif layer_dict["layer_type"] == "bn":
                decoder.append(nn.BatchNorm1d(num_features=layer_dict["num_features"]))
            elif layer_dict["layer_type"] == "relu":
                decoder.append(nn.ReLU())
            elif layer_dict["layer_type"] == "relu6":
                decoder.append(nn.ReLU6())
            elif layer_dict["layer_type"] == "dropout":
                decoder.append(nn.Dropout1d(p=layer_dict.get("p", self.dropout)))

        self.decoder = nn.Sequential(*decoder)

        if self.frozen_decoder:
            self.freeze_decoder()

    def initialize(self):
        self.initialize_encoder()
        self.initialize_decoder()

    def forward(self, inputs: torch.Tensor):
        outputs = self.encoder(inputs)
        outputs = self.decoder(outputs)
        return outputs

    def prepare_train(self):
        if self.frozen_encoder:
            self.encoder.eval()
        else:
            self.encoder.train()

        if self.frozen_decoder:
            self.decoder.eval()
        else:
            self.decoder.train()

    def prepare_eval(self):
        self.encoder.eval()
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


class TorchClassIncrementalModel(TorchClassificationModel):
    def __init__(self, **kwargs):
        super().__init__(num_classes=None, **kwargs)

    def initialize(self):
        self.initialize_encoder()

    def update_decoder(self, task_id: int, task: str | list[str]):
        num_new_classes = len(task) if isinstance(task, list) else 1
        if task_id == 0:
            self.num_classes = num_new_classes
            self.initialize_decoder()
        else:
            self.num_classes += num_new_classes

            fc = nn.Linear(
                in_features=self.encoder.output_size,
                out_features=self.num_classes,
            )

            nb_output = self.decoder.fc.out_features
            weight = copy.deepcopy(self.decoder.fc.weight.data)
            bias = copy.deepcopy(self.decoder.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

            del self.decoder.fc
            self.decoder.fc = fc
        self.decoder.to(config.device)


class TorchBottleneckClassificationModel(TorchClassificationModel):
    def forward(self, inputs: torch.Tensor):
        outputs = self.encoder(inputs)
        outputs = self.bottleneck(outputs)
        outputs = self.decoder(outputs)
        return outputs

    def initialize_decoder(self):
        self.decoder = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)), nn.Flatten())

    def prepare_keys_initialization(self):
        self.encoder.eval()
        self.bottleneck.train()
        self.decoder.eval()

    def prepare_train(self):
        super().prepare_train()
        self.bottleneck.eval()

    def prepare_eval(self):
        super().prepare_eval()
        self.bottleneck.eval()


class TorchBottleneckClassIncrementalModel(TorchClassIncrementalModel):
    def forward(self, inputs: torch.Tensor):
        outputs = self.encoder(inputs)
        outputs = self.bottleneck(outputs)
        outputs = self.decoder(outputs)
        return outputs

    def prepare_keys_initialization(self):
        self.encoder.eval()
        self.bottleneck.train()
        self.decoder.eval()

    def prepare_train(self):
        super().prepare_train()
        self.bottleneck.eval()

    def prepare_eval(self):
        super().prepare_eval()
        self.bottleneck.eval()


class TorchMertClassificationModel(TorchClassificationModel):
    def __init__(
        self,
        encoder: dict[str, str | dict] = {
            "name": "MertEncoder",
            "args": {
                "pretrained": True,
            },
        },
        frozen_encoder: bool = True,
        **kwargs
    ):
        super().__init__(encoder=encoder, frozen_encoder=frozen_encoder, **kwargs)

    def initialize_encoder(self):
        if self.frozen_encoder:
            self.freeze_encoder()

    def initialize_decoder(self):
        in_features = self.encoder.output_size
        self.decoder = MertClassificationDecoder(
            conv1_dict=self.head_config[0],
            in_features=in_features,
            num_classes=self.num_classes,
        )
        if self.frozen_decoder:
            self.freeze_decoder()


class TorchMertClassIncrementalModel(TorchClassIncrementalModel):
    def __init__(
        self,
        encoder: dict[str, str | dict] = {
            "name": "MertEncoder",
            "args": {
                "pretrained": True,
            },
        },
        frozen_encoder: bool = True,
        **kwargs
    ):
        super().__init__(encoder=encoder, frozen_encoder=frozen_encoder, **kwargs)

    def initialize_encoder(self):
        if self.frozen_encoder:
            self.freeze_encoder()

    def initialize_decoder(self):
        in_features = self.encoder.output_size
        self.decoder = MertClassificationDecoder(
            conv1_dict=self.head_config[0],
            in_features=in_features,
            num_classes=self.num_classes,
        )
        if self.frozen_decoder:
            self.freeze_decoder()


class TorchMertBottleneckClassIncrementalModel(TorchMertClassIncrementalModel):
    def forward(self, inputs: torch.Tensor):
        outputs = self.encoder(inputs)
        outputs = self.bottleneck(outputs)
        outputs = self.decoder(outputs)
        return outputs

    def prepare_keys_initialization(self):
        self.encoder.eval()
        self.bottleneck.train()
        self.decoder.eval()

    def prepare_train(self):
        super().prepare_train()
        self.bottleneck.eval()

    def prepare_eval(self):
        super().prepare_eval()
        self.bottleneck.eval()
