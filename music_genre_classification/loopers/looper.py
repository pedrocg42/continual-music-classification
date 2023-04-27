from abc import ABC, abstractmethod


class Looper(ABC):
    def __init__(self, **kwargs):
        self.experiment_name = None
        self.early_stopping = False

    @abstractmethod
    def train_batch(self):
        pass

    @abstractmethod
    def train_epoch(self, epoch: int):
        pass

    @abstractmethod
    def val_batch(self):
        pass

    @abstractmethod
    def val_epoch(self, epoch: int):
        pass
