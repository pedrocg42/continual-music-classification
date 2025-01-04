from abc import ABC, abstractmethod


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
