import os
from abc import ABC, abstractmethod

import torch

import config
from models import TrainModel


class MusicGenderClassificationModelSaver(ABC):
    def __init__(self, output_folder: str = config.models_path, **kwargs):
        self.model: TrainModel = None
        self.output_folder = output_folder

    def configure(self, model: TrainModel, experiment_name: str):
        self.model = model
        self.experiment_name = experiment_name

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(config.models_path, f"{self.experiment_name}.pt"),
        )

    def save_model_cl(self, task: int = None, cv: int | None = None):
        output_file = os.path.join(config.models_path, f"{self.experiment_name}")
        output_file += f"__cv_{cv}" if cv is not None else ""
        output_file += f"__task_{task}" if task is not None else ""
        output_file += f".pt"
        torch.save(self.model.state_dict(), output_file)

    def load_model(self, model: TrainModel, checkpoint_file: str):
        return NotImplementedError

    def check_if_already_exported(self, **kwargs) -> bool:
        return NotImplementedError
