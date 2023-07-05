from abc import ABC

from loguru import logger

from music_genre_classification.loopers import LooperFactory


class Trainer(ABC):
    def __init__(
        self,
        looper: dict,
        num_epochs: int,
        batch_size: int,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "F1 Score",
        debug: bool = False,
    ):
        self.looper = LooperFactory.build(looper)
        self.num_epochs = num_epochs if debug is False else 2
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.best_metric = 0
        self.patience_epochs = 0

        # Debug
        self.debug = debug
        self.looper.debug = debug
        self.max_epochs = 1

    def configure_cv(self, cross_val_id: int):
        self.best_metric = 0
        self.patience_epochs = 0
        self.looper.initialize_model()
        self.looper.configure_task(cross_val_id=cross_val_id)
        self.looper.log_start()

    def early_stopping(self, metrics: dict):
        if metrics[self.early_stopping_metric] > self.best_metric:
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
        self.looper.configure_experiment(experiment_name)
        for cross_val_id in range(num_cross_val_splits):
            self.configure_cv(cross_val_id)
            if self.looper.model_saver.model_exists():
                logger.info(f"Model already exists for cross_val_id {cross_val_id}")
                continue
            for epoch in range(self.num_epochs):
                results = self.looper.train_epoch(epoch=epoch)
                metrics = self.looper.extract_metrics(results)
                self.looper.log_metrics(metrics, epoch)
                results = self.looper.val_epoch(epoch)
                metrics = self.looper.extract_metrics(results, mode="val")
                self.looper.log_metrics(metrics, epoch, mode="val")
                if self.early_stopping(metrics):
                    break
