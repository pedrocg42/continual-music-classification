# Found filtered dataset from https://github.com/coreyker/dnn-mgr/tree/master/gtzan
import os
from glob import glob

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import config
from music_genre_classification.train_data_sources.mert_genre_classification_dataset import (
    MertGenreClassificationDataset,
)
from music_genre_classification.train_data_sources.train_data_source import (
    TrainDataSource,
)

np.random.seed(config.seed)


SINGERS = [
    "female1",
    "female2",
    "male1",
    "male2",
    "female3",
    "female4",
    "male3",
    "male4",
    "female5",
    "female6",
    "male5",
    "male6",
    "female7",
    "female8",
    "male7",
    "male8",
    "female9",
    "male9",
    "male10",
    "male11",
]


class VocalSetSingerDataSource(TrainDataSource):
    def __init__(
        self,
        split: str,
        num_cross_val_splits: int = 5,
        singer: list[str] = SINGERS,
        is_eval: bool = False,
        chunk_length: float = 5.0,
        **kwargs,
    ):
        self.name = "VocalSetSinger"
        self.dataset_path = os.path.join(config.dataset_path, "VocalSet", "FULL")
        self.singer = singer
        self.singer_to_index = {singer: i for i, singer in enumerate(self.singer)}
        self.index_to_singer = {i: singer for i, singer in enumerate(self.singer)}

        # Split parameters
        self.split = split
        self.num_cross_val_splits = num_cross_val_splits
        self.is_eval = is_eval

        # Audio parameters
        self.sample_rate = 44100
        self.song_length = 6  # Average time per clip, useful for training
        self.chunk_length = chunk_length

        self._get_songs()

    def _get_songs(self):
        # Read annotations
        self.songs = np.array(
            glob(os.path.join(self.dataset_path, "*", "*", "*", "*.wav"))
        )

        song_labels = np.array(
            [os.path.normpath(song).split(os.sep)[-4] for song in self.songs]
        )
        self.labels = np.array([self.singer_to_index[label] for label in song_labels])

        # Shuffling
        idx = np.arange(len(self.songs))
        np.random.shuffle(idx)
        self.songs = self.songs[idx]
        self.labels = self.labels[idx]

        # Split
        self.cross_val_split()

    def cross_val_split(self, tasks: list[list[str]] = 0):
        # Split
        self.songs_splits = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.labels_splits = {
            "train": [],
            "val": [],
            "test": [],
        }
        for index in self.index_to_singer.keys():
            songs_singer = self.songs[self.labels == index]
            labels_singer = self.labels[self.labels == index]
            split_size = len(songs_singer) // self.num_cross_val_splits
            songs_splits = [
                songs_singer[int(i * split_size) : int((i + 1) * split_size)]
                for i in range(self.num_cross_val_splits)
            ]
            labels_splits = [
                labels_singer[int(i * split_size) : int((i + 1) * split_size)]
                for i in range(self.num_cross_val_splits)
            ]

            # Get train, val and test splits
            test_set_songs = songs_splits.pop(tasks)
            val_set_songs = songs_splits.pop(-1)
            train_set_songs = np.concatenate(songs_splits)
            self.songs_splits["train"].append(train_set_songs)
            self.songs_splits["val"].append(val_set_songs)
            self.songs_splits["test"].append(test_set_songs)

            test_set_labels = labels_splits.pop(tasks)
            val_set_labels = labels_splits.pop(-1)
            train_set_labels = np.concatenate(labels_splits)
            self.labels_splits["train"].append(train_set_labels)
            self.labels_splits["val"].append(val_set_labels)
            self.labels_splits["test"].append(test_set_labels)

        self.songs_splits["train"] = np.concatenate(self.songs_splits["train"])
        self.songs_splits["val"] = np.concatenate(self.songs_splits["val"])
        self.songs_splits["test"] = np.concatenate(self.songs_splits["test"])
        self.labels_splits["train"] = np.concatenate(self.labels_splits["train"])
        self.labels_splits["val"] = np.concatenate(self.labels_splits["val"])
        self.labels_splits["test"] = np.concatenate(self.labels_splits["test"])

    def get_dataset(
        self,
        task: str | list[str] = None,
        tasks: list[list[str]] = 0,
        memory_dataset: Dataset = None,
        is_eval: bool | None = None,
    ) -> Dataset:
        self.cross_val_split(tasks=tasks)

        songs = self.songs_splits[self.split]
        labels = self.labels_splits[self.split]

        if task is not None and task != "all":
            if isinstance(task, str):
                songs = songs[labels == self.singer_to_index[task]]
                labels = labels[labels == self.singer_to_index[task]]
            elif isinstance(task, list):
                task = [self.singer_to_index[singer] for singer in task]
                songs = songs[np.isin(labels, task)]
                labels = labels[np.isin(labels, task)]

        dataset = MertGenreClassificationDataset(
            songs=songs,
            labels=labels,
            is_eval=self.is_eval if is_eval is None else is_eval,
            song_length=self.song_length,
            audio_length=self.chunk_length,
            input_sample_rate=self.sample_rate,
        )

        if memory_dataset is not None:
            dataset = ConcatDataset([dataset, memory_dataset])

        return dataset

    def get_dataloader(
        self,
        task: list[str] | str = None,
        tasks: list[list[str]] = 0,
        batch_size: int = 32,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        dataset = self.get_dataset(task=task, tasks=tasks, **kwargs)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True if (self.split == "train") else False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader
