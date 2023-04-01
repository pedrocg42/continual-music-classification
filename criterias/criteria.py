from abc import ABC, abstractmethod
from typing import Any


class Criteria(ABC):
    @abstractmethod
    def calculate_loss(self, inputs: Any, targets: Any):
        return NotImplementedError("Criteria must implement calculate_loss method")

    def __call__(self, *args, **kwargs):
        return self.calculate_loss(*args, **kwargs)
