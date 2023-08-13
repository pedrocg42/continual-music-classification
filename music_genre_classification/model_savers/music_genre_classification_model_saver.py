import os
from abc import ABC

import torch

import config
from music_genre_classification.models import TrainModel


class MusicGenreClassificationModelSaver(ABC):
    def __init__(self, models_folder: str = config.models_path, **kwargs):
        self.model: TrainModel = None
        self.models_folder = models_folder

    def configure(
        self,
        model: TrainModel,
        experiment_name: str,
        tasks: list[list[str]] = None,
        task: str | list[str] = None,
        task_id: int = None,
    ):
        self.model = model

        self.experiment_name = experiment_name
        self.output_folder = os.path.join(self.models_folder, self.experiment_name)
        self.create_output_folder()

        self.tasks = tasks
        self.task = "-".join(task) if isinstance(task, list) else task
        self.task_id = task_id
        self.output_path = self.build_output_path(self.task_id)

    def create_output_folder(self):
        os.makedirs(self.output_folder, exist_ok=True)

    def build_output_path(self, task_id: int = None):
        output_path = os.path.join(self.output_folder, f"{self.experiment_name}")
        output_path += f"__task_{task_id}" if task_id is not None else ""
        output_path += f".pt"
        return output_path

    def save_model(self, output_path: str = None):
        if output_path is None:
            output_path = self.output_path
        torch.save(self.model.state_dict(), output_path)

    def load_model(self, output_path: str = None):
        if output_path is None:
            output_path = self.output_path
        self.model.load_state_dict(torch.load(output_path))

    def load_task_model(self, task_id: int):
        output_path = self.build_output_path(self.tasks, task_id)
        self.load_model(output_path)

    def check_if_already_exported(self, **kwargs) -> bool:
        return NotImplementedError

    def model_exists(self) -> bool:
        return os.path.exists(self.output_path)
