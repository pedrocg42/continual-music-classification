from loguru import logger

from music_genre_classification.trainers.continual_learning_trainer import (
    ContinualLearningTrainer,
)


class DkvbContinualLearningTrainer(ContinualLearningTrainer):
    def __init__(
        self,
        epochs_keys_init: int | None = None,
        freeze_decoder_after_first_episode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.epochs_keys_init = epochs_keys_init
        self.freeze_decoder_after_first_episode = freeze_decoder_after_first_episode

    def configure_cv(self, cross_val_id: int):
        self.looper.initialize_model()

    def configure_task(
        self, cross_val_id: int, task: str | list[str], continual_learning: bool = False
    ):
        self.best_metric = 0
        self.patience_epochs = 0
        if not continual_learning:
            self.looper.initialize_model()
        self.looper.configure_task(cross_val_id=cross_val_id, task=task)
        self.looper.log_start()

    def initialize_keys(self):
        for epoch in range(self.epochs_keys_init):
            self.looper.key_init_epoch(epoch=epoch)

    def train(self, experiment_name: str, num_cross_val_splits: int = 1):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.looper.configure_experiment(experiment_name, self.batch_size)
        for cross_val_id in range(num_cross_val_splits):
            if self.debug and cross_val_id > 0:
                break
            self.configure_cv(cross_val_id)
            self.looper.log_start()
            for task_num, task in enumerate(self.tasks):
                self.configure_task(cross_val_id, task)
                if self.looper.model_saver.model_exists():
                    logger.info(
                        f"Model already exists for cross_val_id {cross_val_id} and task {task}"
                    )
                    continue
                if self.epochs_keys_init is not None and task_num == 0:
                    self.initialize_keys()
                if self.freeze_decoder_after_first_episode and task_num == 1:
                    logger.info("Freezing decoder")
                    self.looper.model.freeze_decoder()
                for epoch in range(self.num_epochs):
                    results = self.looper.train_epoch(epoch=epoch)
                    metrics = self.looper.extract_metrics(results)
                    self.looper.log_metrics(metrics, epoch)
                    results = self.looper.val_epoch(epoch)
                    metrics = self.looper.extract_metrics(results, mode="val")
                    self.looper.log_metrics(metrics, epoch, mode="val")
                    if self.early_stopping(metrics, epoch):
                        break
