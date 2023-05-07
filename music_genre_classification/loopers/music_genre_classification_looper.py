import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import config
from music_genre_classification.loopers import Looper


class MusicGenreClassificationLooper(Looper):
    def configure_experiment(self, experiment_name: str, batch_size: int):
        self.experiment_name = experiment_name
        self.batch_size = batch_size

    def initialize_model(self):
        # Configure model
        self.model.initialize()
        self.model.to(config.device)

        # Configure with model and experiment name
        self.optimizer.configure(self.model.parameters())

        # Move to device
        self.train_data_transform.to(config.device)
        self.val_data_transform.to(config.device)

    def configure_task(self, cross_val_id: int, task: str = None):
        # Configure data loaders
        self.train_data_loader = self.train_data_source.get_dataloader(
            cross_val_id=cross_val_id, task=task, batch_size=self.batch_size
        )
        self.val_data_loader = self.val_data_source.get_dataset(
            cross_val_id=cross_val_id, task=task
        )

        # Configure output components
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            cross_val_id=cross_val_id,
            task=task,
        )
        self.experiment_tracker.configure_task(
            experiment_name=self.experiment_name, cross_val_id=cross_val_id, task=task
        )

    def log_start(self):
        print(self.model)
        logger.info(
            f"> > Total parameters: {sum(param.numel() for param in self.model.parameters())}"
        )

    def train_epoch(self, epoch: int):
        logger.info(f"Training epoch {epoch + 1}")
        self.model.prepare_train()
        results_epoch = []
        pbar = tqdm(self.train_data_loader, colour="green")
        for i, (waveforms, labels) in enumerate(pbar):
            if self.debug and i > self.max_steps:
                break
            results_epoch.append(self.train_batch(waveforms, labels))
            self.update_pbar(pbar, results_epoch)
        return results_epoch

    def train_batch(self, waveforms: torch.Tensor, labels: torch.Tensor):
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
        self.model.prepare_eval()
        results_epoch = []
        pbar = tqdm(self.val_data_loader, colour="magenta")
        for i, (waveforms, labels) in enumerate(pbar):
            if self.debug and i > self.max_steps:
                break
            results_epoch.append(self.val_batch(waveforms, labels))
            self.update_pbar(pbar, results_epoch)
        return results_epoch

    @torch.no_grad()
    def val_batch(self, waveforms: torch.Tensor, labels: torch.Tensor):
        waveforms = waveforms.to(config.device)
        labels = labels.to(config.device)

        # Inference
        transformed = self.val_data_transform(waveforms)
        preds = self.model(transformed)

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
        for metric_name, metric in self.metrics.items():
            metrics_results[metric_name] = metric(preds, labels).item()

        return metrics_results

    def log_metrics(
        self, metrics_results: dict[str, float], epoch: int, mode: str = "train"
    ):
        for metric_name, metric_result in metrics_results.items():
            self.experiment_tracker.log_metric(
                f"{metric_name.title()}/{mode}", metric_result, epoch
            )

    def update_pbar(self, pbar: tqdm, results_epoch: dict[str, float]):
        pbar.set_postfix(
            {"loss": np.mean([result_epoch["loss"] for result_epoch in results_epoch])}
        )
