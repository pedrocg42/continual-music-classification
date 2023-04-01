import os

from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import config
from experiment_tracker.experiment_tracker import ExperimentTracker


class TensorboardExperimentTracker(ExperimentTracker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        logger.info(" > Creating TensorBoard writer")
        self.writer = None

    def configure(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(
            log_dir=os.path.join(config.logs_path, self.experiment_name)
        )

    def log_metric(self, metric_name: str, metric: float, epoch: int):
        self.writer.add_scalar(metric_name, metric, epoch)
