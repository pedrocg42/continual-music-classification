import torch
from torch import Tensor, nn

import config
from music_genre_classification.loopers.music_genre_classification_looper import (
    MusicGenreClassificationLooper,
)


class iCaRLMusicGenreClassificationLooper(MusicGenreClassificationLooper):
    def __init__(self, T: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.T = T

        # Added after first training task
        self.old_model = None
        self.known_classes = []

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
        preds = model(transformed)

        # Compute loss
        clf_loss = self.criteria(preds, labels)
        if self.old_model is not None:
            # After first training task
            loss_kd = self.knowledge_distilattion_loss(
                preds[:, : len(self.known_classes)],
                self.old_model(transformed),
                self.T,
            )
            loss = clf_loss + loss_kd
        else:
            loss = clf_loss
        loss.backward()

        # Adjust weights
        self.optimizer.step()

        return dict(
            loss=loss.detach().cpu().item(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )

    @staticmethod
    def knowledge_distilattion_loss(pred: Tensor, soft: Tensor, T: float = 2.0):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
