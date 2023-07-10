import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import config
from music_genre_classification.loopers.music_genre_classification_looper import (
    MusicGenreClassificationLooper,
)


class GemMusicGenreClassificationLooper(MusicGenreClassificationLooper):
    def train_batch(self, waveforms: torch.Tensor, labels: torch.Tensor):
        self.optimizer.before_training_iteration(
            self.model, self.criteria, self.train_data_transform, self.task_id
        )
        waveforms = waveforms.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        # Zero gradient before every batch
        self.optimizer.zero_grad()

        # Inference
        transformed = self.train_data_transform(waveforms, augment=True)
        preds = self.model(transformed)

        # Compute loss
        loss = self.criteria(preds, labels)
        loss.backward()

        self.optimizer.after_backward(self.model, self.task_id)

        # Adjust weights
        self.optimizer.step()

        return dict(
            loss=loss.detach().cpu().item(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )
