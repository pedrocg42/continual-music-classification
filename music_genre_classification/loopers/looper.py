from abc import ABC, abstractmethod

from music_genre_classification.criterias import CriteriaFactory
from music_genre_classification.experiment_trackers import ExperimentTrackerFactory
from music_genre_classification.metrics import MetricsFactory
from music_genre_classification.model_savers import ModelSaverFactory
from music_genre_classification.models import TrainModelFactory
from music_genre_classification.optimizers import OptimizerFactory
from music_genre_classification.train_data_sources import TrainDataSourceFactory
from music_genre_classification.train_data_transforms import TrainDataTransformFactory


class Looper(ABC):
    def __init__(
        self,
        train_data_source: dict,
        val_data_source: dict,
        train_data_transform: dict | None,
        val_data_transform: dict | None,
        train_model: dict,
        criteria: dict,
        optimizer: dict,
        experiment_tracker: dict,
        model_saver: dict,
        metrics: dict,
    ) -> None:
        self.train_data_source = TrainDataSourceFactory.build(train_data_source)
        self.train_data_loader = None
        self.train_data_transform = TrainDataTransformFactory.build(
            train_data_transform
        )

        self.val_data_source = TrainDataSourceFactory.build(val_data_source)
        self.val_data_loader = None
        self.val_data_transform = TrainDataTransformFactory.build(val_data_transform)

        self.model = TrainModelFactory.build(train_model)

        # Configure optimizer and criteria (loss function)
        self.optimizer = OptimizerFactory.build(optimizer)
        self.criteria = CriteriaFactory.build(criteria)

        # Metrics
        self.metrics = MetricsFactory.build(metrics)

        # Experiment tracker
        self.experiment_tracker = ExperimentTrackerFactory.build(experiment_tracker)

        # Model saver
        self.model_saver = ModelSaverFactory.build(model_saver)

        # Debug
        self.debug = False
        self.max_steps = 5

    @abstractmethod
    def train_batch(self):
        pass

    @abstractmethod
    def train_epoch(self, epoch: int):
        pass

    @abstractmethod
    def val_batch(self):
        pass

    @abstractmethod
    def val_epoch(self, epoch: int):
        pass
