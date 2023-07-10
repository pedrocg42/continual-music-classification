from loguru import logger

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
        self.looper.initialize_model()

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
            self.looper.initialize_model()
        self.looper.configure_task(
            cross_val_id=cross_val_id, task_id=task_id, task=task
        )

    def early_stopping(self, metrics: dict, epoch: int = 0):
        if metrics[self.early_stopping_metric] > self.best_metric or epoch == 0:
            self.best_metric = metrics[self.early_stopping_metric]
            self.patience_epochs = 0
            self.looper.model_saver.save_model()
        else:
            self.patience_epochs += 1

        if self.patience_epochs >= self.early_stopping_patience:
            logger.info("Early stopping")
            return True
        return False

    def train(self, experiment_name: str, num_cross_val_splits: int = 1):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.looper.configure_experiment(experiment_name, self.batch_size)
        for cross_val_id in range(num_cross_val_splits):
            if cross_val_id > 0 or self.debug and cross_val_id > 0:
                break
            logger.info(f"Started training of {cross_val_id=}")
            self.configure_cv(cross_val_id)
            self.looper.log_start()
            for task_id, task in enumerate(self.tasks):
                logger.info(f"Started training process of {task_id=} {task=}")
                self.configure_task(cross_val_id, task_id, task)
                if self.looper.model_saver.model_exists():
                    logger.info(
                        f"Model already exists for cross_val_id {cross_val_id} and task {task}"
                    )
                    continue
                for epoch in range(self.num_epochs):
                    # Train
                    results = self.looper.train_epoch(epoch=epoch)
                    metrics = self.looper.extract_metrics(results)
                    self.looper.log_metrics(metrics, epoch)
                    # Val
                    results = self.looper.val_epoch(epoch)
                    metrics = self.looper.extract_metrics(results)
                    self.looper.log_metrics(metrics, epoch, mode="val")
                    if self.early_stopping(metrics, epoch):
                        break
