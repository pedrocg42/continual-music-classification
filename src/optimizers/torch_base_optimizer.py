import torch.optim

from src.optimizers.optimizer import Optimizer


class TorchBaseOptimizer(Optimizer):
    optimizer: torch.optim.Optimizer = None

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def get_lr(self) -> float:
        lrl = [param_group["lr"] for param_group in self.optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        return lr

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
