import torch.nn
from torch import device

from music_genre_classification.models.train_model import TrainModel


class TorchBaseModel(TrainModel):
    model: torch.nn.Module = None
    input_size: tuple[int, int, int] = None

    def parameters(self):
        return self.model.parameters()

    def to(self, device: device):
        self.model = self.model.to(device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.model.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        self.model.load_state_dict(state_dict, strict=strict)

    def __repr__(self) -> str:
        return self.model.__repr__()
