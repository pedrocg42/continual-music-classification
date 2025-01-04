from typing import Iterable

from torch import Tensor
from torch.optim import Adam

from src.optimizers.torch_base_optimizer import (
    TorchBaseOptimizer,
)


class TorchAdamOptimizer(TorchBaseOptimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.optimizer = None

    def configure(self, parameters: Iterable[Tensor] | Iterable[dict], **kwargs):
        self.optimizer = Adam(
            parameters,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def step(self):
        self.optimizer.step()
