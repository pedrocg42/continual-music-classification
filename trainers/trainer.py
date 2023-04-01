from abc import ABC
from loopers import Looper
from loguru import logger


class Trainer(ABC):
    def __init__(self, looper: Looper, num_epochs: int):
        self.looper = looper
        self.num_epochs = num_epochs

    def train(self, experiment_name: str):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.looper.configure(experiment_name=experiment_name)
        self.looper.log_start()
        for epoch in range(self.num_epochs):
            results = self.looper.train_epoch(epoch=epoch)
            self.looper.log_train_epoch(results, epoch)
            results = self.looper.val_epoch(epoch)
            self.looper.log_val_epoch(results, epoch)
            self.looper.save_model()
            if self.looper.early_stopping:
                break
