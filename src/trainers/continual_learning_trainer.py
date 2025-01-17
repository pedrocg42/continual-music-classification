import numpy as np
import torch
from loguru import logger

from src.trainers.trainer import Trainer


class ContinualLearningTrainer(Trainer):
    def __init__(
        self,
        tasks: list[str | list[str]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tasks = tasks

    def configure_task(
        self,
        task_id: int,
        task: list[str] | str = None,
        continual_learning: bool = True,
    ):
        self.looper.task_id = task_id
        self.looper.task = task

        self.best_metric = 0
        self.patience_epochs = 0

        if not continual_learning:
            self.initialize_model()

        # Configure data loaders
        self.train_data_loader = self.train_data_source.get_dataloader(
            tasks=self.tasks, task=task, batch_size=self.batch_size
        )
        self.val_data_loader = self.val_data_source.get_dataset(tasks=self.tasks, task=task, is_eval=True)

        # Configure model saver and load model if exists
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            task_id=task_id,
            task=task,
        )
        if task_id > 0:
            self.model_saver.load_task_model(task_id - 1)

        # Configure experiment tracker
        self.experiment_tracker.configure_task(
            experiment_name=self.experiment_name,
            task_id=task_id,
            task=task,
        )

    def train(self, experiment_name: str):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.configure_experiment(experiment_name, self.batch_size)
        self.log_start()
        for task_id, task in enumerate(self.tasks):
            self.configure_task(task_id=task_id, task=task)
            if self.model_saver.model_exists():
                logger.info(f"Model already exists task {task}")
                continue
            logger.info(f"Starting training of task {task}")
            for epoch in range(self.num_epochs):
                early_stopping = self.train_epoch(epoch)
                if early_stopping:
                    break

    def extract_metrics(self, results_epoch: list[dict]):
        preds = torch.vstack([results_batch["preds"] for results_batch in results_epoch])
        labels = torch.hstack([results_batch["labels"] for results_batch in results_epoch])
        metrics_results = {}
        metrics_results["loss"] = np.array([results_batch["loss"] for results_batch in results_epoch]).mean()
        for metric_name, metric in self.metrics.items():
            metric.num_classes = self.model.num_classes
            metrics_results[metric_name] = metric(preds, labels).item()

        return metrics_results
