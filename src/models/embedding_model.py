import copy

import torch
import torch.nn as nn

import config
from src.models.encoders import EncoderFactory


class TorchEmbeddingModel(nn.Module):
    def __init__(self, encoder: dict, average_hidden: bool = True, **kwargs):
        super().__init__()
        self.encoder = EncoderFactory.build(encoder)
        self.average_hidden = average_hidden

        self.register_buffer(
            "reference_embeddings", torch.zeros(0, self.encoder.output_size)
        )
        self.initialize()

    def initialize_encoder(self):
        self.freeze_encoder()

    def initialize(self):
        self.initialize_encoder()

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(inputs)
        if self.average_hidden:
            embeddings = torch.mean(embeddings, dim=1)
        return embeddings

    def match_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        similarities = -torch.cdist(embeddings[None], self.reference_embeddings[None])[
            0
        ]
        return similarities

    def forward(self, inputs: torch.Tensor):
        embeddings = self.forward_features(inputs)
        similarities = self.match_embeddings(embeddings)
        return similarities

    def update_references(self, embeddings: torch.Tensor, labels: torch.Tensor):
        unique_labels = torch.unique(labels)
        for unique_label in unique_labels:
            class_embeddings = embeddings[labels == unique_label]
            mean_embedding = torch.mean(class_embeddings, dim=0, keepdims=True)
            self.reference_embeddings = torch.cat(
                [
                    self.reference_embeddings,
                    mean_embedding.to(config.device),
                ]
            )

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
