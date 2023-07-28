import torch

from music_genre_classification.models.classification_model import (
    TorchMertClassIncrementalModel,
)


class TorchL2PClassIncrementalModel(TorchMertClassIncrementalModel):
    def forward(self, inputs: torch.Tensor):
        outputs, key_loss = self.encoder(inputs)
        outputs = self.decoder(outputs)
        return outputs, key_loss
