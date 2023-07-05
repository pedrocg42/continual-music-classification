from loguru import logger

from music_genre_classification.trainers.class_incremental_learning_trainer import (
    ClassIncrementalLearningTrainer,
)


class DkvbContinualLearningTrainer(ClassIncrementalLearningTrainer):
    def __init__(
        self,
        epochs_keys_init: int | None = None,
        freeze_decoder_after_first_episode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if epochs_keys_init is not None:
            self.epochs_keys_init = 2 if self.debug else epochs_keys_init
        else:
            self.epochs_keys_init = epochs_keys_init
        self.freeze_decoder_after_first_episode = freeze_decoder_after_first_episode

    def initialize_keys(self):
        logger.info("Initializing keys")
        for epoch in range(self.epochs_keys_init):
            self.looper.key_init_epoch(epoch=epoch)

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
                    continue
                if self.epochs_keys_init is not None and task_id == 0:
                    self.initialize_keys()
                if self.freeze_decoder_after_first_episode and task_id > 0:
                    logger.info("Freezing decoder")
                    self.looper.model.freeze_decoder()
                for epoch in range(self.num_epochs):
                    if self.debug and epoch > self.max_epochs:
                        break
                    results = self.looper.train_epoch(epoch=epoch)
                    metrics = self.looper.extract_metrics(results)
                    self.looper.log_metrics(metrics, epoch)
                    results = self.looper.val_epoch(epoch)
                    metrics = self.looper.extract_metrics(results)
                    self.looper.log_metrics(metrics, epoch, mode="val")
                    if self.early_stopping(metrics, epoch):
                        break
