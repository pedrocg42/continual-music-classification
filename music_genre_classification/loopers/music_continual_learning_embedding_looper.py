from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config


class MusicContinualLearningEmbeddingLooper(ABC):
    def __init__(self):
        # Debug
        self.debug = False
        self.max_steps = 5

        self.task_id = None
        self.task = None

    @torch.no_grad()
    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        data_loader: DataLoader | Dataset,
        data_transform: nn.Module,
    ) -> dict[str, torch.Tensor]:
        logger.info(f"Training epoch")
        model.prepare_train()
        results_epoch = []
        pbar = tqdm(
            data_loader,
            colour="green",
            total=self.max_steps if self.debug else len(data_loader),
        )
        for i, (inputs, labels) in enumerate(pbar):
            if self.debug and i > self.max_steps:
                break
            results_epoch.append(
                self.train_batch(model, inputs, labels, data_transform)
            )
        embeddings = torch.concat([epoch_dict["preds"] for epoch_dict in results_epoch])
        labels = torch.concat([epoch_dict["labels"] for epoch_dict in results_epoch])

        model.update_references(embeddings, labels)

        # Extract similarities
        preds = model.match_embeddings(embeddings.to(config.device))

        return dict(preds=preds.cpu(), labels=labels)

    def train_batch(
        self,
        model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        data_transform: nn.Module,
    ):
        inputs = inputs.to(config.device, non_blocking=True)

        # Inference
        transformed = data_transform(inputs, augment=True)
        preds = model.forward_features(transformed)

        return dict(
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )

    @torch.no_grad()
    def val_epoch(
        self,
        epoch: int,
        model: nn.Module,
        data_loader: DataLoader | Dataset,
        data_transform: nn.Module,
    ):
        logger.info(f"Validation epoch {epoch + 1}")
        model.prepare_eval()
        results_epoch = []
        pbar = tqdm(
            data_loader,
            colour="magenta",
            total=self.max_steps if self.debug else len(data_loader),
        )
        for i, (inputs, labels) in enumerate(pbar):
            if self.debug and i > self.max_steps:
                break
            results_epoch.append(self.val_batch(model, inputs, labels, data_transform))
        preds = torch.concat([epoch_dict["preds"] for epoch_dict in results_epoch])
        labels = torch.concat([epoch_dict["labels"] for epoch_dict in results_epoch])
        return dict(preds=preds, labels=labels)

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
        preds = model(transformed)

        return dict(
            preds=preds.cpu(),
            labels=labels.cpu(),
        )
