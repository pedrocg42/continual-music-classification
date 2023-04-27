from abc import ABC, abstractmethod
from typing import Any


class TrainModel(ABC):
    def __init__(self, **kwargs):
        self.experiment_name = None

    @abstractmethod
    def inference(self, input: Any):
        return NotImplementedError("TrainModel must implement call method")

    def __call__(self, *args, **kwargs):
        return self.inference(*args, **kwargs)
