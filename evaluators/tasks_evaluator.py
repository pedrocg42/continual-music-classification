from abc import ABC

import torch
from loguru import logger
from torchmetrics import Metric
from tqdm import tqdm

import config
from experiment_tracker import ExperimentTracker
from model_savers import ModelSaver
from models import TrainModel
from train_data_sources import TrainDataSource
from train_data_transforms import TrainDataTransform


class TasksEvaluator(ABC):
    def __init__(
        self,
        tasks: list[str],
        model: TrainModel,
        model_saver: ModelSaver,
        data_source: TrainDataSource,
        data_transform: TrainDataTransform,
        metrics: dict[str, Metric],
        experiment_tracker: ExperimentTracker,
    ):
        # Basic information
        self.experiment_name = None
        self.experiment_type = None
        self.experiment_subtype = None
        self.num_cross_val_splits = None
        self.tasks = tasks

        # Components
        self.model = model
        self.model_saver = model_saver
        self.data_source = data_source
        self.data_loader = None
        self.data_transform = data_transform
        self.metrics = metrics
        self.experiment_tracker = experiment_tracker

    def configure(
        self,
        experiment_name: str,
        experiment_type: str,
        experiment_subtype: str,
        num_cross_val_splits: int,
    ):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.experiment_subtype = experiment_subtype
        self.num_cross_val_splits = num_cross_val_splits

        self.experiment_tracker.configure(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            experiment_subtype=experiment_subtype,
            dataset_name=self.data_source.name,
        )

    def configure_task(self, cross_val_id: int, task: str = None):
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            cross_val_id=cross_val_id,
            task=task,
        )
        self.model_saver.load_model()
        self.model.to(config.device)

    def predict(self, data_loader) -> list[dict]:
        self.model.eval()
        results = []
        for waveforms, labels in tqdm(data_loader, colour="green"):
            waveforms = waveforms.to(config.device)
            labels = labels.to(config.device)

            # Inference
            spectrograms = self.data_transform(waveforms)
            preds = self.model(spectrograms.repeat(1, 3, 1, 1))

            # For each song we select the most repeated class
            # preds = preds.detach().cpu().numpy()
            # unique_values, unique_counts = np.unique(preds, return_counts=True)

            results.append(
                dict(
                    preds=preds.detach().cpu(),
                    labels=labels.detach().cpu(),
                )
            )
        return results

    def extract_metrics(self, results: list[dict]) -> dict:
        metrics = {}
        preds = torch.vstack([result["preds"] for result in results])
        labels = torch.hstack([result["labels"] for result in results])
        for metric_name, metric in self.metrics.items():
            metric_result = metric(preds, labels)
            metrics[metric_name] = metric_result
        return metrics

    def evaluate(
        self,
        experiment_name: str,
        experiment_type: str,
        experiment_subtype: str,
        num_cross_val_splits: int,
    ):
        logger.info(f"Started evaluation process of experiment {experiment_name}")
        self.configure(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            experiment_subtype=experiment_subtype,
            num_cross_val_splits=num_cross_val_splits,
        )

        for cross_val_id in range(self.num_cross_val_splits):
            self.configure_task(cross_val_id=cross_val_id)
            self.experiment_tracker.configure_task(
                cross_val_id=cross_val_id, task_name="all"
            )
            # Extracting results on the whole dataset
            data_loader = self.data_source.get_dataset(cross_val_id=cross_val_id)
            results = self.predict(data_loader)
            metrics = self.extract_metrics(results)
            self.experiment_tracker.log_metrics(metrics)
            # Extracting results for each task
            for task in tqdm(self.tasks, colour="green"):
                data_loader = self.data_source.get_dataset(
                    cross_val_id=cross_val_id, task=task
                )
                results = self.predict(data_loader)
                metrics = self.extract_metrics(results)
                self.experiment_tracker.configure_task(
                    cross_val_id=cross_val_id, task_name=task
                )
                self.experiment_tracker.log_metrics(metrics)