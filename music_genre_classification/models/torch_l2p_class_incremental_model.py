import torch

from music_genre_classification.models.classification_model import (
    TorchMertClassIncrementalModel,
)
from music_genre_classification.models.encoders.mert_encoder_l2p import MertEncoderL2P


class TorchL2PClassIncrementalModel(TorchMertClassIncrementalModel):
    encoder: MertEncoderL2P

    def forward(self, inputs: torch.Tensor):
        outputs, key_loss = self.encoder(inputs)
        outputs = self.decoder(outputs)
        return outputs, key_loss

    def prepare_train(self):
        self.encoder.freeze_encoder()
        self.encoder.prompt_pool.train()

        if self.frozen_decoder:
            self.decoder.eval()
        else:
            self.decoder.train()

    def prepare_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def freeze_encoder(self):
        self.encoder.freeze_encoder()
        self.frozen_encoder = True
