from loguru import logger

from music_genre_classification.trainers.class_incremental_learning_trainer import (
    ClassIncrementalLearningTrainer,
)


class EwcContinualLearningTrainer(ClassIncrementalLearningTrainer):
    def train(self, experiment_name: str, num_cross_val_splits: int = 1):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.looper.configure_experiment(experiment_name, self.batch_size)
        for cross_val_id in range(num_cross_val_splits):
            if cross_val_id > 0 or self.debug and cross_val_id > 0:
                break
            self.configure_cv(cross_val_id)
            self.looper.log_start()
            for task_id, task in enumerate(self.tasks):
                self.configure_task(cross_val_id, task_id, task)
                if self.looper.model_saver.model_exists():
                    logger.info(
                        f"Model already exists for cross_val_id {cross_val_id} and task {task}"
                    )
                    self.looper.optimizer.after_training_task(
                        self.looper.model,
                        self.looper.train_data_loader,
                        self.looper.train_data_transform,
                        self.looper.criteria,
                        task_id,
                    )
                    continue
                logger.info(f"Starting training of task {task}")
                for epoch in range(self.num_epochs):
                    results = self.looper.train_epoch(epoch=epoch)
                    metrics = self.looper.extract_metrics(results)
                    self.looper.log_metrics(metrics, epoch)
                    results = self.looper.val_epoch(epoch)
                    metrics = self.looper.extract_metrics(results)
                    self.looper.log_metrics(metrics, epoch, mode="val")
                    if self.early_stopping(metrics, epoch):
                        break
                self.looper.optimizer.after_training_task(
                    self.looper.model,
                    self.looper.train_data_loader,
                    self.looper.train_data_transform,
                    self.looper.criteria,
                    task_id,
                )
