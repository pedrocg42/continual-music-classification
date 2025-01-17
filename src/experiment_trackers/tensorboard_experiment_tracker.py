import os

from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import config
from src.experiment_trackers.experiment_tracker import (
    ExperimentTracker,
)


class TensorboardExperimentTracker(ExperimentTracker):
    def __init__(self, logs_folder: str = config.logs_path, **kwargs) -> None:
        super().__init__(**kwargs)

        self.logs_folder = logs_folder
        self.writer = None

    def configure_task(
        self,
        experiment_name: str,
        task_id: int,
        task: str | list[str] = None,
    ):
        self.experiment_name = experiment_name
        self.task_id = task_id
        self.task = "-".join(task) if isinstance(task, list) else task
        self.build_model_name()

        logger.info(" > Creating TensorBoard writer")
        self.writer = SummaryWriter(log_dir=os.path.join(self.logs_folder, self.model_name))

    def build_model_name(
        self,
    ):
        self.model_name = f"{self.experiment_name}"
        self.model_name += f"__task_{self.task_id}" if self.task_id is not None else ""

    def log_metric(self, metric_name: str, metric: float, epoch: int):
        self.writer.add_scalar(metric_name, metric, epoch)
