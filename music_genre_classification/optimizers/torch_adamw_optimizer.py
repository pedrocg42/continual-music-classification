from typing import Iterable

from torch import Tensor
from torch.optim import AdamW

from music_genre_classification.optimizers.torch_base_optimizer import (
    TorchBaseOptimizer,
)


class TorchAdamWOptimizer(TorchBaseOptimizer):
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
        self.optimizer = AdamW(
            parameters,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def step(self):
        self.optimizer.step()
