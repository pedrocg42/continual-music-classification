from abc import ABC, abstractmethod

from loguru import logger

import config
from src.experiment_trackers import ExperimentTrackerFactory
from src.loopers import LooperFactory
from src.metrics import MetricsFactory
from src.model_savers import ModelSaverFactory
from src.models import TrainModelFactory
from src.train_data_sources import TrainDataSourceFactory
from src.train_data_transforms import TrainDataTransformFactory


class Trainer(ABC):
    def __init__(
        self,
        looper: dict,
        num_epochs: int,
        batch_size: int,
        early_stopping_patience: int,
        early_stopping_metric: str,
        train_data_source: dict,
        val_data_source: dict,
        train_data_transform: dict | None,
        val_data_transform: dict | None,
        train_model: dict,
        experiment_tracker: dict,
        model_saver: dict,
        metrics_config: dict,
        debug: bool = False,
    ):
        self.experiment_name = None

        # General config
        self.num_epochs = num_epochs if debug is False else 2
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.best_metric = 0
        self.patience_epochs = 0

        # Data
        self.train_data_source = TrainDataSourceFactory.build(train_data_source)
        self.train_data_loader = None
        self.train_data_transform = TrainDataTransformFactory.build(
            train_data_transform
        )

        self.val_data_source = TrainDataSourceFactory.build(val_data_source)
        self.val_data_loader = None
        self.val_data_transform = TrainDataTransformFactory.build(val_data_transform)

        # Model
        self.model = TrainModelFactory.build(train_model)

        # Metrics
        self.metrics_config = metrics_config
        self.metrics = MetricsFactory.build(self.metrics_config)

        # Experiment tracker
        self.experiment_tracker = ExperimentTrackerFactory.build(experiment_tracker)

        # Model saver
        self.model_saver = ModelSaverFactory.build(model_saver)

        # Looper
        self.looper = LooperFactory.build(looper)

        # Debug
        self.debug = debug
        self.looper.debug = debug

    def configure_experiment(self, experiment_name: str, batch_size: int):
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.initialize_model()

    def initialize_model(self):
        # Configure model
        self.model.initialize()
        self.model.to(config.device)

        # Configure with model and experiment name
        self.looper.optimizer.configure(self.model.parameters())

        # Move to device
        self.train_data_transform.to(config.device)
        self.val_data_transform.to(config.device)

    def log_start(self):
        # print(self.model)
        logger.info(
            f"> > Total parameters: {sum(param.numel() for param in self.model.parameters())}"
        )

    def early_stopping(self, metrics: dict, epoch: int = 0):
        if metrics[self.early_stopping_metric] > self.best_metric or epoch == 0:
            self.best_metric = metrics[self.early_stopping_metric]
            self.patience_epochs = 0
            self.model_saver.save_model()
        else:
            self.patience_epochs += 1

        if self.patience_epochs >= self.early_stopping_patience:
            logger.info("Early stopping")
            return True
        return False

    def train_epoch(self, epoch: int):
        results = self.looper.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_data_loader,
            data_transform=self.train_data_transform,
        )
        metrics = self.extract_metrics(results)
        self.log_metrics(metrics, epoch)
        results = self.looper.val_epoch(
            epoch,
            model=self.model,
            data_loader=self.val_data_loader,
            data_transform=self.val_data_transform,
        )
        metrics = self.extract_metrics(results)
        self.log_metrics(metrics, epoch, mode="val")
        return self.early_stopping(metrics, epoch)

    @abstractmethod
    def extract_metrics(self, results_epoch: list[dict]) -> dict[str, float]:
        pass

    def log_metrics(
        self, metrics_results: dict[str, float], epoch: int, mode: str = "train"
    ):
        for metric_name, metric_result in metrics_results.items():
            self.experiment_tracker.log_metric(
                f"{metric_name.title()}/{mode}", metric_result, epoch
            )
