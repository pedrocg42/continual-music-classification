import torch
from loguru import logger
from torchmetrics import Metric
from tqdm import tqdm

import config
from music_genre_classification.experiment_tracker import ExperimentTracker
from music_genre_classification.model_savers import ModelSaver
from music_genre_classification.models import TrainModel
from music_genre_classification.train_data_sources import TrainDataSource
from music_genre_classification.train_data_transforms import TrainDataTransform


class ContinualLearningTasksEvaluatorV2:
    def __init__(
        self,
        train_tasks: list[str],
        test_tasks: list[str],
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
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

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

    def configure_task(self, cross_val_id: int, task_num: int, task: str):
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            cross_val_id=cross_val_id,
            task=task,
        )
        self.model_saver.load_model()
        self.model.to(config.device)
        self.experiment_tracker.configure_task(
            cross_val_id=cross_val_id,
            train_task_number=task_num,
            train_task_name=task,
        )

    def predict(self, data_loader) -> list[dict]:
        self.model.eval()
        results = []
        for waveforms, labels in tqdm(data_loader, colour="green"):
            waveforms = waveforms.to(config.device)

            # Inference
            spectrograms = self.data_transform(waveforms)
            preds = self.model(spectrograms)

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
            metrics[metric_name] = metric_result.item()
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
            for task_num, task in enumerate(self.train_tasks):
                logger.info(f"Started evaluation of task {task}")
                self.configure_task(
                    cross_val_id=cross_val_id, task_num=task_num, task=task
                )
                # Extracting results for all tasks
                data_loader = self.data_source.get_dataset(cross_val_id=cross_val_id)
                results = self.predict(data_loader)
                metrics = self.extract_metrics(results)
                self.experiment_tracker.log_task_metrics(metrics, "all")
                # Extracting results per task
                for task_test in self.test_tasks:
                    data_loader = self.data_source.get_dataset(
                        cross_val_id=cross_val_id, task=task_test
                    )
                    results = self.predict(data_loader)
                    metrics = self.extract_metrics(results)
                    self.experiment_tracker.log_task_metrics(metrics, task_test)
