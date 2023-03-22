from abc import ABC, abstractmethod


class ExperimentTracker(ABC):
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
