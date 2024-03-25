from torch import Tensor
from music_genre_classification.models.classification_model import (
    TorchClassIncrementalModel,
    TorchClassificationModel,
)
from music_genre_classification.models.decoders.mert_classification_decoder import (
    MertClassificationDecoder,
)


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
        self.decoder = MertClassificationDecoder()(
            conv1_dict=self.head_config[0],
            in_features=in_features,
            num_classes=self.num_classes,
        )
        if self.frozen_decoder:
            self.freeze_decoder()


class TorchMertBottleneckClassIncrementalModel(TorchMertClassIncrementalModel):
    def forward(self, inputs: Tensor):
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
