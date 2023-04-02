import os

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import config
from criterias import Criteria
from experiment_tracker import ExperimentTracker
from loopers import Looper
from model_savers import ModelSaver
from models import TrainModel
from optimizers import Optimizer
from train_data_sources import TrainDataSource
from train_data_transforms import TrainDataTransform


class MusicGenderClassificationLooper(Looper):
    def __init__(
        self,
        train_data_source: TrainDataSource,
        val_data_source: TrainDataSource,
        train_data_transform: TrainDataTransform | None,
        val_data_transform: TrainDataTransform | None,
        train_model: TrainModel,
        criteria: Criteria,
        optimizer: Optimizer,
        experiment_tracker: ExperimentTracker,
        model_saver: ModelSaver,
        metrics: dict,
    ) -> None:
        super().__init__()
        logger.info(f"Training experiment: {self.experiment_name}")

        self.train_data_source = train_data_source
        self.train_data_transform = train_data_transform

        self.val_data_source = val_data_source
        self.val_data_transform = val_data_transform

        self.model = train_model

        # Configure optimizer and criteria (loss function)
        self.optimizer = optimizer
        self.criteria = criteria

        # Metrics
        self.metrics = metrics

        # Experiment tracker
        self.experiment_tracker = experiment_tracker

        # Model saver
        self.model_saver = model_saver

    def configure(self, **kwargs):
        self.optimizer.configure(self.model.parameters(), **kwargs)
        self.experiment_tracker.configure(**kwargs)
        self.train_data_transform.to(config.device)
        self.val_data_transform.to(config.device)
        self.model.to(config.device)
        self.model_saver.configure(self.model, **kwargs)

    def log_start(self):
        print(self.model)
        logger.info(
            f"> > Total parameters: {sum(param.numel() for param in self.model.parameters())}"
        )

    def train_epoch(self, epoch: int):
        logger.info(f"Training epoch {epoch + 1}")
        self.model.train()
        results_epoch = []
        for waveforms, labels in tqdm(self.train_data_source, colour="green"):
            results_epoch.append(self.train_batch(waveforms, labels))
        return results_epoch

    def train_batch(self, waveforms: torch.Tensor, labels: torch.Tensor):
        waveforms = waveforms.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        # Zero gradient before every batch
        self.optimizer.zero_grad()

        # Inference
        spectrograms = self.train_data_transform(waveforms, augment=True)
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
        logger.info(f"Validation epoch {epoch + 1}")
        self.model.eval()
        results_epoch = []
        pbar = tqdm(self.val_data_source, colour="green")
        for waveforms, labels in tqdm(self.val_data_source, colour="green"):
            results_epoch.append(self.val_batch(waveforms, labels))
            pbar.set_postfix(
                {
                    "loss": np.mean(
                        [result_epoch["loss"].numpy() for result_epoch in results_epoch]
                    )
                }
            )
        return results_epoch

    @torch.no_grad()
    def val_batch(self, waveforms: torch.Tensor, labels: torch.Tensor):
        waveforms = waveforms.to(config.device)
        labels = labels.to(config.device)

        # Inference
        spectrograms = self.val_data_transform(waveforms)
        preds = self.model(spectrograms.repeat(1, 3, 1, 1))

        # Compute loss
        loss = self.criteria(preds, labels)

        return dict(
            loss=loss.detach().cpu().item(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
        )

    def extract_metrics(self, results_epoch: list[dict], mode: str = "train"):
        preds = torch.vstack(
            [results_batch["preds"] for results_batch in results_epoch]
        )
        labels = torch.hstack(
            [results_batch["labels"] for results_batch in results_epoch]
        )
        metrics_results = {}
        metrics_results["loss"] = np.array(
            [results_batch["loss"] for results_batch in results_epoch]
        ).mean()
        for metric_name, metric in self.metrics[mode].items():
            metrics_results[metric_name] = metric(preds, labels).cpu().item().numpy()

        return metrics_results

    def log_metrics(
        self, metrics_results: dict[str, float], epoch: int, mode: str = "train"
    ):
        for metric_name, metric_result in metrics_results.items():
            self.experiment_tracker.log_metric(
                f"{metric_name.title()}/{mode}", metric_result, epoch
            )
