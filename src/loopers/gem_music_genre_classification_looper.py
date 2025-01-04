import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import config
from src.loopers.music_genre_classification_looper import (
    MusicGenreClassificationLooper,
)


class GemMusicGenreClassificationLooper(MusicGenreClassificationLooper):
    def train_batch(
        self,
        model: torch.nn.Module,
        waveforms: torch.Tensor,
        labels: torch.Tensor,
        data_transform: torch.nn.Module,
    ):
        self.optimizer.before_training_iteration(model, self.criteria, data_transform)
        waveforms = waveforms.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        # Zero gradient before every batch
        self.optimizer.zero_grad()

        # Inference
        transformed = data_transform(waveforms, augment=True)
        preds = model(transformed)

        # Compute loss
        loss = self.criteria(preds, labels)
        loss.backward()

        self.optimizer.after_backward(model, self.task_id)

        # Adjust weights
        self.optimizer.step()

        return dict(
            loss=loss.detach().cpu().item(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )
