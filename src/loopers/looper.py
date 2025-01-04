import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from src.criterias import CriteriaFactory
from src.optimizers import OptimizerFactory


class Looper:
    def __init__(
        self,
        criteria: dict,
        optimizer: dict,
    ) -> None:
        # Configure optimizer and criteria (loss function)
        self.optimizer = OptimizerFactory.build(optimizer)
        self.criteria = CriteriaFactory.build(criteria)

        # Debug
        self.debug = False
        self.max_steps = 5

    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        data_loader: DataLoader | Dataset,
        data_transform: nn.Module,
    ):
        logger.info(f"Training epoch {epoch + 1}")
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
            if len(inputs) == 1:
                logger.warning("Last batch with length 1 is not allowed")
                break
            results_epoch.append(self.train_batch(model, inputs, labels, data_transform))
            self.update_pbar(pbar, results_epoch)
        return results_epoch

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
        loss = self.criteria(preds, labels)
        loss.backward()

        # Adjust weights
        self.optimizer.step()

        return dict(
            loss=loss.detach().cpu().item(),
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
            self.update_pbar(pbar, results_epoch)
        return results_epoch

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

        # Compute loss
        loss = self.criteria(preds, labels)

        return dict(
            loss=loss.detach().cpu().item(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )

    def update_pbar(self, pbar: tqdm, results_epoch: dict[str, float]):
        pbar.set_postfix({"loss": np.mean([result_epoch["loss"] for result_epoch in results_epoch])})
