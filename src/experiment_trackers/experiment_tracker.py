from abc import ABC, abstractmethod


class ExperimentTracker(ABC):
    def __init__(self, **kwargs) -> None:
        self.experiment_name = None

    @abstractmethod
    def configure(self):
        raise NotImplementedError("ExperimentTracker must implement configure method")

    @abstractmethod
    def log_metric(self):
        raise NotImplementedError("ExperimentTracker must implement log_metric method")

    @abstractmethod
    def log_metrics(self):
        raise NotImplementedError("ExperimentTracker must implement log_metrics method")
