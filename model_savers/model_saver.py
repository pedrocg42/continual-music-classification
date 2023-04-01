from abc import ABC, abstractmethod

from models import TrainModel


class TrainModelSaver(ABC):
    def __init__(
        self, model: TrainModel, output_folder: str, final_export: bool = True, **kwargs
    ):
        self.model: TrainModel = model
        self.output_folder: str = output_folder

    @abstractmethod
    def save_model(self, epoch: int, metric: float = None):
        return NotImplementedError

    @abstractmethod
    def load_model(self, model: TrainModel, checkpoint_file: str):
        return NotImplementedError

    @abstractmethod
    def check_if_already_exported(self, **kwargs) -> bool:
        return NotImplementedError
