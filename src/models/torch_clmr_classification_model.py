from src.models.classification_model import (
    TorchClassificationModel,
    TorchClassIncrementalModel,
)
from src.models.decoders.clrm_classification_decoder import (
    ClmrClassificationDecoder,
)


class TorchClmrClassificationModel(TorchClassificationModel):
    def __init__(
        self,
        encoder: dict[str, str | dict] = None,
        frozen_encoder: bool = True,
        **kwargs,
    ):
        if encoder is None:
            encoder = {"name": "MertEncoder", "args": {"pretrained": True}}
        super().__init__(encoder=encoder, frozen_encoder=frozen_encoder, **kwargs)

    def initialize_encoder(self):
        if self.frozen_encoder:
            self.freeze_encoder()

    def initialize_decoder(self):
        in_features = self.encoder.output_size
        self.decoder = ClmrClassificationDecoder(
            in_features=in_features,
            num_classes=self.num_classes,
        )
        if self.frozen_decoder:
            self.freeze_decoder()


class TorchClmrClassIncrementalModel(TorchClassIncrementalModel):
    def __init__(
        self,
        encoder: dict[str, str | dict] = None,
        frozen_encoder: bool = True,
        **kwargs,
    ):
        if encoder is None:
            encoder = {"name": "ClmrEncoder", "args": {"pretrained": True}}
        super().__init__(encoder=encoder, frozen_encoder=frozen_encoder, **kwargs)

    def initialize_encoder(self):
        if self.frozen_encoder:
            self.freeze_encoder()

    def initialize_decoder(self):
        in_features = self.encoder.output_size
        self.decoder = ClmrClassificationDecoder(
            in_features=in_features,
            num_classes=self.num_classes,
        )
        if self.frozen_decoder:
            self.freeze_decoder()
