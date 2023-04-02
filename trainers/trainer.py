from abc import ABC

from loguru import logger

from loopers import Looper


class Trainer(ABC):
    def __init__(
        self,
        looper: Looper,
        num_epochs: int,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "F1 Score",
    ):
        self.looper = looper
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.best_metric = 0
        self.patience_epochs = 0

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

    def train(self, experiment_name: str):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.looper.configure(experiment_name=experiment_name)
        self.looper.log_start()
        for epoch in range(self.num_epochs):
            results = self.looper.train_epoch(epoch=epoch)
            metrics = self.looper.extract_metrics(results)
            self.looper.log_metrics(metrics, epoch)
            results = self.looper.val_epoch(epoch)
            metrics = self.looper.extract_metrics(results, mode="val")
            self.looper.log_metrics(metrics, epoch, mode="val")
            if self.early_stopping(metrics):
                break
