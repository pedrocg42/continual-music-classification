import torch
from torch import Tensor, nn

import config
from src.loopers.music_genre_classification_looper import (
    MusicGenreClassificationLooper,
)


class L2PMusicGenreClassificationLooper(MusicGenreClassificationLooper):
    def __init__(self, lamb: float, **kwargs):
        super().__init__(**kwargs)
        self.lamb = lamb

    def train_batch(
        self,
        model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        data_transform: nn.Module,
    ):
        inputs = inputs.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        # Zero gradient before every batch
        self.optimizer.zero_grad()

        # Inference
        transformed = data_transform(inputs, augment=True)
        preds, key_loss = model(transformed)

        # Compute loss
        loss = self.criteria(preds, labels)
        total_loss = loss + self.lamb * key_loss
        total_loss.backward()

        # Adjust weights
        self.optimizer.step()

        return dict(
            loss=total_loss.detach().cpu().item(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )

    @torch.no_grad()
    def val_batch(
        self,
        model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        data_transform: nn.Module,
    ):
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        # Inference
        transformed = data_transform(inputs)
        preds, key_loss = model(transformed)

        # Compute loss
        loss = self.criteria(preds, labels)
        total_loss = loss + self.lamb * key_loss

        return dict(
            loss=total_loss.detach().cpu().item(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )
