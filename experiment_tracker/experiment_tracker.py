from abc import ABC, abstractmethod


class ExperimentTracker(ABC):
    def __init__(self, **kwargs) -> None:
        self.experiment_name = None

    @abstractmethod
    def configure(self):
        pass

    @abstractmethod
    def log_metric(self):
        pass

    def log_metrics(self):
        pass
