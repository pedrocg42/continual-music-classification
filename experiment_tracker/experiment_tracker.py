from abc import ABC, abstractmethod


class ExperimentTracker(ABC):
    def __init__(self, **kwargs) -> None:
        self.experiment_name = None

    def configure(self):
        pass

    def log_metric(self):
        pass

    def log_metrics(self):
        pass
