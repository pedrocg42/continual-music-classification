from loguru import logger

from music_genre_classification.metrics import MetricsFactory
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

        if epochs_keys_init is not None:
            self.epochs_keys_init = 2 if self.debug else epochs_keys_init
        else:
            self.epochs_keys_init = epochs_keys_init
        self.freeze_decoder_after_first_episode = freeze_decoder_after_first_episode

    def configure_task(
        self,
        cross_val_id: int,
        task_id: int,
        task: str | list[str],
        **kwargs,
    ):
        super().configure_task(cross_val_id, task_id, task, **kwargs)
        self.model.update_bottleneck(task_id, task)
        self.looper.optimizer.configure(self.model.parameters())

        # Updating metrics
        for metric_config in self.metrics_config:
            metric_config["args"].update({"num_classes": self.model.num_classes})
        self.metrics = MetricsFactory.build(self.metrics_config)

    def initialize_keys(self):
        logger.info("Initializing keys")
        for epoch in range(self.epochs_keys_init):
            self.looper.key_init_epoch(epoch, self.model, self.train_data_loader)

    def train(self, experiment_name: str, num_cross_val_splits: int = 1):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.configure_experiment(experiment_name, self.batch_size)
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
                if task_id == 0:
                    self.initialize_keys()
                if self.freeze_decoder_after_first_episode and task_id > 0:
                    logger.info("Freezing decoder")
                    self.model.freeze_decoder()
                for epoch in range(self.num_epochs):
                    early_stopping = self.train_epoch(epoch)
                    if early_stopping:
                        break
