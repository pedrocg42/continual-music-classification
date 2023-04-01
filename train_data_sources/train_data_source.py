from abc import ABC, abstractmethod


class TrainDataSource(ABC):
    @abstractmethod
    def __iter__(self):
        return NotImplementedError("TrainDataSource must implement __iter__ method")

    @abstractmethod
    def __len__(self):
        return NotImplementedError("TrainDataSource must implement __len__ method")

    @abstractmethod
    def shuffle(self):
        return NotImplementedError("TrainDataSource must implement shuffle method")
