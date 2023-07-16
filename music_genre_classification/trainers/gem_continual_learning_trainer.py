from loguru import logger

from music_genre_classification.trainers.class_incremental_learning_trainer import (
    ClassIncrementalLearningTrainer,
)


class GemContinualLearningTrainer(ClassIncrementalLearningTrainer):
    def train(self, experiment_name: str, num_cross_val_splits: int = 1):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.configure_experiment(experiment_name, self.batch_size)
        for cross_val_id in range(num_cross_val_splits):
            if cross_val_id > 0 or self.debug and cross_val_id > 0:
                break
            self.configure_cv(cross_val_id)
            self.log_start()
            for task_id, task in enumerate(self.tasks):
                self.configure_task(cross_val_id, task_id, task)
                if self.model_saver.model_exists():
                    logger.info(
                        f"Model already exists for cross_val_id {cross_val_id} and task {task}"
                    )
                    self.looper.optimizer.after_training_task(
                        self.train_data_loader,
                        task_id,
                    )
                    continue
                logger.info(f"Starting training of task {task}")
                for epoch in range(self.num_epochs):
                    early_stopping = self.train_epoch(epoch)
                    if early_stopping:
                        break
                self.looper.optimizer.after_training_task(
                    self.train_data_loader,
                    task_id,
                )
