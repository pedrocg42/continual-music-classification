from abc import ABC, abstractmethod

from music_genre_classification.models import TrainModel


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
    def model_exists(self, **kwargs) -> bool:
        pass
