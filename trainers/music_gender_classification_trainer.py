import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils import data
from tqdm import tqdm

import config
from experiment_tracker import ExperimentTracker


class MusicGenderClassificationTrainer:
    def __init__(
        self,
        dataset_class: data.Dataset,
        extraction_pipeline: nn.Module,
        criteria_class: nn.modules.loss._Loss,
        optimizer_class: torch.optim.Optimizer,
        experiment_tracker_class: ExperimentTracker,
        metrics: dict,
        architecture: nn.Module,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        **experiment,
    ) -> None:
        self.experiment_name = experiment["experiment_name"]
        logger.info(f"Training experiment: {self.experiment_name}")

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.train_dataset = dataset_class(split="train", **experiment)
        self.train_dataloader = self.train_dataset.get_dataloader(
            batch_size=self.batch_size, num_workers=2
        )

        self.val_dataset = dataset_class(split="val", **experiment)

        self.extraction_pipeline = extraction_pipeline.to(config.preprocess_device)

        # Building the model
        logger.info(" > Building model")
        self.model = architecture(**experiment)
        self.model.to(config.device)
        print(self.model)
        logger.info(
            f"> > Total parameters: {sum(param.numel() for param in self.model.parameters())}"
        )

        # Configure optimizer and loss function
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)
        self.criteria = criteria_class()

        # Metrics
        self.metrics = metrics

        # Experiment tracker
        self.experiment_tracker = experiment_tracker_class(**experiment)

    def train(
        self,
    ):
        for epoch in range(self.num_epochs):
            results_epoch = self.train_epoch(epoch)
            self.log_train_epoch(results_epoch, epoch)
            self.val_epoch(epoch)
            self.log_val_epoch(results_epoch, epoch)

        self.save_model()

    def train_epoch(self, epoch: int):
        logger.info(f"Training epoch {epoch}")
        self.model.train()
        results_epoch = []
        for waveforms, labels in tqdm(self.train_dataloader, colour="green"):
            results_epoch.append(self.train_batch(waveforms, labels))
        return results_epoch

    def train_batch(self, waveforms: torch.Tensor, labels: torch.Tensor):
        waveforms = waveforms.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        # Zero gradient before every batch
        self.optimizer.zero_grad()

        # Inference
        spectrograms = self.extraction_pipeline(waveforms, augment=True)
        preds = self.model(spectrograms.repeat(1, 3, 1, 1))

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
    def val_epoch(self, epoch: int):
        logger.info(f"Training epoch {epoch}")
        self.model.eval()
        results_epoch = []
        for waveforms, labels in tqdm(self.val_dataset, colour="green"):
            results_epoch.append(self.val_batch(waveforms, labels))
        return results_epoch

    @torch.no_grad()
    def val_batch(self, waveforms: torch.Tensor, labels: torch.Tensor):
        waveforms = waveforms.to(config.device)
        labels = labels.to(config.device)

        # Inference
        spectrograms = self.extraction_pipeline(waveforms)
        preds = self.model(spectrograms.repeat(1, 3, 1, 1))

        # Compute loss
        loss = self.criteria(preds, labels)

        return dict(
            loss=loss.detach().cpu().item(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )

    def log_train_epoch(self, results_epoch: list[dict], epoch: int):
        avg_loss = np.array(
            [results_batch["loss"] for results_batch in results_epoch]
        ).mean()
        self.experiment_tracker.log_metric("Loss/Train", avg_loss, epoch)

        if self.metrics.get("train", None) is not None:
            preds = torch.vstack(
                [results_batch["preds"] for results_batch in results_epoch]
            )
            labels = torch.hstack(
                [results_batch["labels"] for results_batch in results_epoch]
            )
            for metric_name, metric in self.metrics["train"].items():
                metric_result = metric(preds, labels)
                self.experiment_tracker.log_metric(
                    f"{metric_name}/Train", metric_result, epoch
                )

    def log_val_epoch(self, results_epoch: list[dict], epoch: int):
        avg_loss = np.array(
            [results_batch["loss"] for results_batch in results_epoch]
        ).mean()
        self.experiment_tracker.log_metric("Loss/Val", avg_loss, epoch)

        if self.metrics.get("val", None) is not None:
            preds = torch.vstack(
                [results_batch["preds"] for results_batch in results_epoch]
            )
            labels = torch.hstack(
                [results_batch["labels"] for results_batch in results_epoch]
            )
            for metric_name, metric in self.metrics["val"].items():
                metric_result = metric(preds, labels)
                self.experiment_tracker.log_metric(
                    f"{metric_name}/Val", metric_result, epoch
                )

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(config.models_path, self.experiment_name),
        )
