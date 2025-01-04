import torch
from loguru import logger

import config
from src.metrics import MetricsFactory
from src.trainers.class_incremental_learning_trainer import (
    ClassIncrementalLearningTrainer,
)


class ContinualLearningTrainerL2Center(ClassIncrementalLearningTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = 0

    def initialize_model(self):
        # Configure model
        self.model.initialize()
        self.model.to(config.device)

        # Move to device
        self.train_data_transform.to(config.device)
        self.val_data_transform.to(config.device)

    def configure_task(
        self,
        task_id: int,
        task: list[str] | str = None,
        continual_learning: bool = True,
    ):
        self.num_classes += len(task)

        self.looper.task_id = task_id
        self.looper.task = task

        if not continual_learning:
            self.initialize_model()

        # Configure data loaders
        self.train_data_loader = self.train_data_source.get_dataset(
            tasks=self.tasks, task=task, is_eval=True
        )
        self.val_data_loader = self.val_data_source.get_dataset(
            tasks=self.tasks, task=task, is_eval=True
        )

        # Configure model saver and load model if exists
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            task_id=task_id,
            task=task,
        )

        # Configure experiment tracker
        self.experiment_tracker.configure_task(
            experiment_name=self.experiment_name,
            task_id=task_id,
            task=task,
        )

        # Updating metrics
        for metric_config in self.metrics_config:
            metric_config["args"].update({"num_classes": self.num_classes})
        self.metrics = MetricsFactory.build(self.metrics_config)

    def train(self, experiment_name: str):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.configure_experiment(experiment_name, self.batch_size)
        self.log_start()
        for task_id, task in enumerate(self.tasks):
            self.configure_task(task_id, task)
            if self.model_saver.model_exists():
                logger.info(f"Model already exists for and task {task}")
                continue
            logger.info(f"Starting training of task {task}")
            self.train_epoch(1)
            self.model_saver.save_model()

    def extract_metrics(self, results_epoch: list[dict]):
        preds = results_epoch["preds"]
        labels = results_epoch["labels"]
        metrics_results = {}
        for metric_name, metric in self.metrics.items():
            metrics_results[metric_name] = metric(preds, labels).item()

        return metrics_results
