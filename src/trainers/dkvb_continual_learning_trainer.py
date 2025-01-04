from loguru import logger

from src.metrics import MetricsFactory
from src.trainers.continual_learning_trainer import (
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
        task_id: int,
        task: list[str] | str | list[str],
        **kwargs,
    ):
        super().configure_task(task_id, task, **kwargs)
        self.model.update_bottleneck(task_id, task)
        self.looper.optimizer.configure(self.model.parameters())

        # Updating metrics
        for metric_config in self.metrics_config:
            metric_config["args"].update({"num_classes": self.model.num_classes})
        self.metrics = MetricsFactory.build(self.metrics_config)

    def initialize_keys(self):
        logger.info("Initializing keys")
        for epoch in range(self.epochs_keys_init):
            self.looper.key_init_epoch(epoch, self.model, self.train_data_loader, self.train_data_transform)

    def train(
        self,
        experiment_name: str,
    ):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.configure_experiment(experiment_name, self.batch_size)
        self.log_start()
        for task_id, task in enumerate(self.tasks):
            logger.info(f"Started training process of {task_id=} {task=}")
            self.configure_task(task_id, task)
            if self.model_saver.model_exists():
                logger.info(f"Model already exists for and task {task}")
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
