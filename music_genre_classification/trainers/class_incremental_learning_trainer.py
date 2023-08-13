import numpy as np
import torch

from music_genre_classification.metrics import MetricsFactory
from music_genre_classification.trainers.continual_learning_trainer import (
    ContinualLearningTrainer,
)


class ClassIncrementalLearningTrainer(ContinualLearningTrainer):
    def configure_task(
        self,
        tasks: list[list[str]],
        task_id: int,
        task: str | list[str],
        **kwargs,
    ):
        super().configure_task(tasks, task_id, task, **kwargs)
        self.model.update_decoder(task_id, task)
        self.looper.optimizer.configure(self.model.parameters())

        # Updating metrics
        for metric_config in self.metrics_config:
            metric_config["args"].update({"num_classes": self.model.num_classes})
        self.metrics = MetricsFactory.build(self.metrics_config)

    def extract_metrics(self, results_epoch: list[dict]):
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
