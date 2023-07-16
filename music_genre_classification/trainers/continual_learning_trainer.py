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
        task: str | list[str],
        continual_learning: bool = True,
    ):
        self.best_metric = 0
        self.patience_epochs = 0
        if not continual_learning:
            self.initialize_model()
        self.looper.configure_task(
            cross_val_id=cross_val_id, task_id=task_id, task=task
        )

    def configure_task(self, cross_val_id: int, task_id: int, task: str = None):
        self.task_id = task_id
        self.task = task

        # Configure data loaders
        self.train_data_loader = self.train_data_source.get_dataloader(
            cross_val_id=cross_val_id, task=task, batch_size=self.batch_size
        )
        self.val_data_loader = self.val_data_source.get_dataset(
            cross_val_id=cross_val_id, task=task
        )

        # Configure output components
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            cross_val_id=cross_val_id,
            task_id=task_id,
            task=task,
        )
        if task_id > 0:
            self.model_saver.load_task_model(task_id - 1)
        self.experiment_tracker.configure_task(
            experiment_name=self.experiment_name,
            cross_val_id=cross_val_id,
            task_id=task_id,
            task=task,
        )

    def train(self, experiment_name: str, num_cross_val_splits: int = 1):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.looper.configure_experiment(experiment_name, self.batch_size)
        for cross_val_id in range(num_cross_val_splits):
            if cross_val_id > 0 or self.debug and cross_val_id > 0:
                break
            logger.info(f"Started training of {cross_val_id=}")
            self.configure_cv(cross_val_id)
            self.log_start()
            for task_id, task in enumerate(self.tasks):
                logger.info(f"Started training process of {task_id=} {task=}")
                self.configure_task(cross_val_id, task_id, task)
                if self.model_saver.model_exists():
                    logger.info(
                        f"Model already exists for cross_val_id {cross_val_id} and task {task}"
                    )
                    continue
                for epoch in range(self.num_epochs):
                    early_stopping = self.train_epoch(epoch)
                    if early_stopping:
                        break
