import numpy as np
import torch
from torch.utils.data import Dataset


class MemoryDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, **kwargs):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        # Get info
        input = self.data[index]
        target = self.targets[index]
        return torch.tensor(input), torch.tensor(target)

    def __len__(self):
        return len(self.data)
