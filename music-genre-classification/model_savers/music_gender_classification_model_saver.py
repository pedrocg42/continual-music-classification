import os
from abc import ABC

import torch

import config
from models import TrainModel


class MusicGenderClassificationModelSaver(ABC):
    def __init__(self, models_folder: str = config.models_path, **kwargs):
        self.model: TrainModel = None
        self.models_folder = models_folder

    def configure(
        self,
        model: TrainModel,
        experiment_name: str,
        cross_val_id: int = None,
        task: str = None,
    ):
        self.model = model

        self.experiment_name = experiment_name
        self.output_folder = os.path.join(self.models_folder, self.experiment_name)
        self.create_output_folder()

        self.cross_val_id = cross_val_id
        self.task = "-".join(task) if isinstance(task, list) else task
        self.build_output_path()

    def create_output_folder(self):
        os.makedirs(self.output_folder, exist_ok=True)

    def build_output_path(self):
        self.output_path = os.path.join(self.output_folder, f"{self.experiment_name}")
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
