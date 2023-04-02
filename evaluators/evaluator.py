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


class Evaluator(ABC):
    def __init__(
        self,
        model: TrainModel,
        model_saver: ModelSaver,
        data_source: TrainDataSource,
        metrics: dict[str, Metric],
        experiment_tracker: ExperimentTracker,
    ):
        self.model = model
        self.model_saver = model_saver
        self.data_source = data_source
        self.metrics = metrics
        self.experiment_tracker = experiment_tracker

    def configure(
        self, experiment_name: str, experiment_type: str, experiment_subtype: str
    ):
        self.model_saver.configure(experiment_name=experiment_name)
        self.experiment_tracker.configure(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            experiment_subtype=experiment_subtype,
        )
        self.model_saver.configure(self.model, experiment_name=experiment_name)
        self.model_saver.load_model()
        self.model.eval()

    def predict(self) -> list[dict]:
        results = []
        for waveforms, labels in tqdm(self.data_source, colour="green"):
            waveforms = waveforms.to(config.device)
            labels = labels.to(config.device)

            # Inference
            spectrograms = self.val_data_transform(waveforms)
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

    def log_metrics(self, metrics: dict):
        for metric_name, metric_value in metrics.items():
            self.experiment_tracker.log_metric(metric_name, metric_value)

    def evaluate(self, experiment_name: str):
        logger.info(f"Started evaluate process of experiment {experiment_name}")
        self.configure(experiment_name=experiment_name)
        results = self.predict()
        metrics = self.extract_metrics(results)
        self.experiment_tracker.log_metrics(metrics)
