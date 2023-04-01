from abc import ABC, abstractmethod
from typing import Any


class Optimizer(ABC):
    @abstractmethod
    def configure(self, **kwargs):
        return NotImplementedError("Optimizer must implement configure")

    @abstractmethod
    def step(self, input: Any, target: Any):
        return NotImplementedError("Criteria must implement step method")

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)
