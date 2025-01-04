from abc import ABC, abstractmethod

from src.base.types import Image, Target


class TrainDataTransform(ABC):
    @abstractmethod
    def transform(self, image: Image, target: Target):
        return NotImplementedError("TrainDataTransform must implement transform method")

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
