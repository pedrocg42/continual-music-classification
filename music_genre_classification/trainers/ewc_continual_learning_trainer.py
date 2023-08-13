from loguru import logger

from music_genre_classification.trainers.class_incremental_learning_trainer import (
    ClassIncrementalLearningTrainer,
)


class EwcContinualLearningTrainer(ClassIncrementalLearningTrainer):
    def train(self, experiment_name: str):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.configure_experiment(experiment_name, self.batch_size)
        self.log_start()
        for task_id, task in enumerate(self.tasks):
            self.configure_task(task_id, task)
            if self.model_saver.model_exists():
                logger.info(f"Model already exists for and task {task}")
                self.looper.optimizer.after_training_task(
                    self.model,
                    self.train_data_loader,
                    self.train_data_transform,
                    self.looper.criteria,
                    task_id,
                )
                continue
            logger.info(f"Starting training of task {task}")
            for epoch in range(self.num_epochs):
                early_stopping = self.train_epoch(epoch)
                if early_stopping:
                    break
            self.looper.optimizer.after_training_task(
                self.model,
                self.train_data_loader,
                self.train_data_transform,
                self.looper.criteria,
                task_id,
            )
