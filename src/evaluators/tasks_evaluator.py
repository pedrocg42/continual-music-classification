import torch
from loguru import logger
from torchmetrics import Metric
from tqdm import tqdm

import config
from src.evaluators.evaluator import Evaluator
from src.experiment_trackers import ExperimentTracker
from src.model_savers import ModelSaver
from src.models import TrainModel
from src.train_data_sources import TrainDataSource
from src.train_data_transforms import TrainDataTransform


class TasksEvaluator(Evaluator):
    def __init__(
        self,
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
    ):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.experiment_subtype = experiment_subtype

        self.experiment_tracker.configure(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            experiment_subtype=experiment_subtype,
            dataset_name=self.data_source.name,
        )

    def configure_task(self, tasks: list[list[str]], task: list[str] | str = None):
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            tasks=tasks,
            task=task,
        )
        self.model_saver.load_model()
        self.model.to(config.device)

    def predict(self, data_loader) -> list[dict]:
        self.model.eval()
        results = []
        for waveforms, labels in tqdm(data_loader, colour="green"):
            waveforms = waveforms.to(config.device)

            # Inference
            spectrograms = self.data_transform(waveforms)
            preds = self.model(spectrograms.repeat(1, 3, 1, 1))

            # For each song we select the most repeated class
            pred = torch.mode(preds.detach().cpu().argmax(dim=1))[0]
            label = labels[0] if len(labels.shape) > 0 else labels

            results.append(
                dict(
                    preds=pred,
                    labels=label,
                )
            )
        return results

    def extract_metrics(self, results: list[dict]) -> dict:
        metrics = {}
        preds = torch.hstack([result["preds"] for result in results])
        labels = torch.hstack([result["labels"] for result in results])
        for metric_name, metric in self.metrics.items():
            metric_result = metric(preds, labels)
            metrics[metric_name] = metric_result.numpy()
        return metrics

    def evaluate(
        self,
        experiment_name: str,
        experiment_type: str,
        experiment_subtype: str,
    ):
        logger.info(f"Started evaluation process of experiment {experiment_name}")
        self.configure(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            experiment_subtype=experiment_subtype,
        )
        self.configure_task()
        self.experiment_tracker.configure_task(tasks=self.tasks)
        # Extracting results
        data_loader = self.data_source.get_dataset(tasks=self.tasks)
        results = self.predict(data_loader)
        metrics = self.extract_metrics(results)
        self.experiment_tracker.log_tasks_metrics(metrics, self.data_source.genres)
