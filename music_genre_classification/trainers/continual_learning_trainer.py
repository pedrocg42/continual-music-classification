from loguru import logger

import config
from music_genre_classification.trainers.trainer import Trainer


class ContinualLearningTrainer(Trainer):
    def __init__(
        self,
        tasks: list[str | list[str]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tasks = tasks

    def configure_cv(self, cross_val_id: int):
        self.initialize_model()

    def configure_task(
        self,
        cross_val_id: int,
        task_id: int,
        task: str = None,
        continual_learning: bool = True,
    ):
        self.task_id = task_id
        self.task = task

        self.best_metric = 0
        self.patience_epochs = 0

        if not continual_learning:
            self.initialize_model()

        # Configure data loaders
        self.train_data_loader = self.train_data_source.get_dataloader(
            cross_val_id=cross_val_id, task=task, batch_size=self.batch_size
        )
        self.val_data_loader = self.val_data_source.get_dataset(
            cross_val_id=cross_val_id, task=task, is_eval=True
        )

        # Configure model saver and load model if exists
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            cross_val_id=cross_val_id,
            task_id=task_id,
            task=task,
        )
        if task_id > 0:
            self.model_saver.load_task_model(task_id - 1)

        # Configure experiment tracker
        self.experiment_tracker.configure_task(
            experiment_name=self.experiment_name,
            cross_val_id=cross_val_id,
            task_id=task_id,
            task=task,
        )
