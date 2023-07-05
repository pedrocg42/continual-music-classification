from loguru import logger

from music_genre_classification.trainers.continual_learning_trainer import (
    ContinualLearningTrainer,
)


class ClassIncrementalLearningTrainer(ContinualLearningTrainer):
    def configure_task(
        self,
        cross_val_id: int,
        task_id: int,
        task: str | list[str],
        **kwargs,
    ):
        super().configure_task(cross_val_id, task_id, task, **kwargs)
        self.looper.model.update_decoder(task_id, task)
