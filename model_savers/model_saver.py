from abc import ABC, abstractmethod

from models import TrainModel


class ModelSaver(ABC):
    @abstractmethod
    def configure(self, model: int, **kwargs):
        return NotImplementedError

    @abstractmethod
    def save_model(self, **kwargs):
        return NotImplementedError

    @abstractmethod
    def load_model(self, **kwargs):
        return NotImplementedError

    @abstractmethod
    def check_if_already_exported(self, **kwargs) -> bool:
        return NotImplementedError
