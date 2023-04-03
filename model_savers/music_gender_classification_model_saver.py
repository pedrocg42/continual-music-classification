import os
from abc import ABC

import torch

import config
from models import TrainModel


class MusicGenderClassificationModelSaver(ABC):
    def __init__(self, output_folder: str = config.models_path, **kwargs):
        self.model: TrainModel = None
        self.output_folder = output_folder

    def configure(
        self,
        model: TrainModel,
        experiment_name: str,
        cross_val_id: int = None,
        task: str = None,
    ):
        self.model = model
        self.experiment_name = experiment_name
        self.cross_val_id = cross_val_id
        self.task = task
        self.build_output_path()

    def build_output_path(self):
        self.output_path = os.path.join(config.models_path, f"{self.experiment_name}")
        self.output_path += (
            f"__cv_{self.cross_val_id}" if self.cross_val_id is not None else ""
        )
        self.output_path += f"__task_{self.task}" if self.task is not None else ""
        self.output_path += f".pt"

    def save_model(self):
        torch.save(self.model.state_dict(), self.output_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.output_path))

    def check_if_already_exported(self, **kwargs) -> bool:
        return NotImplementedError

    def model_exists(self):
        return os.path.exists(self.output_path)
