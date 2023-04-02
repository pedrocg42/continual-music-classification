from abc import ABC, abstractmethod


class TrainDataSource(ABC):
    def __iter__(self):
        return NotImplementedError("TrainDataSource must implement __iter__ method")

    def __len__(self):
        return NotImplementedError("TrainDataSource must implement __len__ method")

    def shuffle(self):
        return NotImplementedError("TrainDataSource must implement shuffle method")
