import copy

import torch
import torch.nn as nn

from music_genre_classification.models.embedding_model import TorchEmbeddingModel
from music_genre_classification.models.encoders import EncoderFactory


class TorchEmbeddingKnnModel(TorchEmbeddingModel):
    def __init__(self, encoder: dict, **kwargs):
        super().__init__()
        self.encoder = EncoderFactory.build(encoder)
        self.register_buffer(
            "reference_embeddings", torch.zeros(0, self.encoder.output_size)
        )
        self.register_buffer("reference_labels", torch.zeros(0))
        self.initialize()

    def match_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        similarities = torch.cdist(
            embeddings[None], self.reference_embeddings[None]
        ).squeeze()
        return similarities

    def forward(self, inputs: torch.Tensor):
        embeddings = self.forward_features(inputs)
        similarities = self.match_embeddings(embeddings)
        return similarities

    def prepare_train(self):
        self.encoder.eval()

    def prepare_eval(self):
        self.encoder.eval()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.frozen_encoder = True

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        self.freeze_encoder()
        return self

    @property
    def num_classes(self) -> int:
        return self.reference_embeddings.shape[0]
