from typing import Any

import torch.nn

from criterias import Criteria


class TorchCrossEntropyCriteria(Criteria):
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def calculate_loss(self, inputs: Any, targets: Any):
        return self.loss(inputs, targets)
