import copy

import numpy as np
import torch
from loguru import logger

from music_genre_classification.trainers.replay_continual_learning_trainer import (
    ReplayContinualLearningTrainer,
)


class iCaRLContinualLearningTrainer(ReplayContinualLearningTrainer):
    def after_training_task(self, cross_val_id: int, task: list[str]):
        super().after_training_task(cross_val_id, task)
        self.looper.old_model = self.model.copy().freeze()
        self.looper.known_classes = self.known_classes
