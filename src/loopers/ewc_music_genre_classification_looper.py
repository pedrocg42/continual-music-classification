import torch

import config
from src.loopers.music_genre_classification_looper import (
    MusicGenreClassificationLooper,
)


class EwcMusicGenreClassificationLooper(MusicGenreClassificationLooper):
    def train_batch(
        self,
        model: torch.nn.Module,
        waveforms: torch.Tensor,
        labels: torch.Tensor,
        data_transform: torch.nn.Module,
    ):
        waveforms = waveforms.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        # Zero gradient before every batch
        self.optimizer.zero_grad()

        # Inference
        transformed = data_transform(waveforms, augment=True)
        preds = model(transformed)

        # Compute loss
        loss = self.criteria(preds, labels)
        self.optimizer.before_backward(model, self.task_id)
        loss.backward()

        # Adjust weights
        self.optimizer.step()

        return dict(
            loss=loss.detach().cpu().item(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )
