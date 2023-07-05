# Found filtered dataset from https://github.com/coreyker/dnn-mgr/tree/master/gtzan
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset

import config
from music_genre_classification.train_data_sources.mert_genre_classification_dataset import (
    MertGenreClassificationDataset,
)
from music_genre_classification.train_data_sources.train_data_source import (
    TrainDataSource,
)
from glob import glob

np.random.seed(config.seed)


GTZAN_GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


class GtzanDataSource(TrainDataSource):
    def __init__(
        self, split: str, num_cross_val_splits: int = 5, is_eval: bool = False, **kwargs
    ):
        self.name = "GTZAN"
        self.dataset_path = os.path.join(config.dataset_path, self.name)
        self.genres = GTZAN_GENRES
        self.genres_to_index = {genre: i for i, genre in enumerate(self.genres)}
        self.index_to_genres = {i: genre for i, genre in enumerate(self.genres)}

        # Split parameters
        self.split = split
        self.num_cross_val_splits = num_cross_val_splits
        self.is_eval = is_eval

        # Audio parameters
        self.sample_rate = 22050
        self.song_length = 30
        self.chunk_length = 3

        self._get_songs()

    def _get_songs(self):
        # Read annotations
        song_list = np.array(
            glob(os.path.join(self.dataset_path, "genres", "*", "*.wav"))
        )
        song_labels = np.array(
            [os.path.basename(song).split(".")[0] for song in song_list]
        )

        # Filter songs
        list_accepted_songs = []
        for split in ["train", "valid", "test"]:
            with open(
                os.path.join(self.dataset_path, f"{split}_filtered.txt"), "r"
            ) as f:
                list_accepted_songs += f.readlines()
        list_accepted_songs = np.array(
            [os.path.basename(song).split(".wav")[0] for song in list_accepted_songs]
        )
        mask = np.array(
            [
                True
                if os.path.basename(song).split(".wav")[0] in list_accepted_songs
                else False
                for song in song_list
            ]
        )
        song_list = song_list[mask]
        song_labels = song_labels[mask]

        # Transform
        self.songs = np.array(
            [
                os.path.join(self.dataset_path, song.split(".")[0], song)
                for song in song_list
            ]
        )
        self.labels = np.array([self.genres_to_index[label] for label in song_labels])

        # Shuffling
        idx = np.arange(len(self.songs))
        np.random.shuffle(idx)
        self.songs = self.songs[idx]
        self.labels = self.labels[idx]

        # Split
        self.cross_val_split()

    def cross_val_split(self, cross_val_id: int = 0):
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
        for index in self.index_to_genres.keys():
            songs_genre = self.songs[self.labels == index]
            labels_genre = self.labels[self.labels == index]
            split_size = len(songs_genre) // self.num_cross_val_splits
            songs_splits = [
                songs_genre[int(i * split_size) : int((i + 1) * split_size)]
                for i in range(self.num_cross_val_splits)
            ]
            labels_splits = [
                labels_genre[int(i * split_size) : int((i + 1) * split_size)]
                for i in range(self.num_cross_val_splits)
            ]

            # Get train, val and test splits
            test_set_songs = songs_splits.pop(cross_val_id)
            val_set_songs = songs_splits.pop(-1)
            train_set_songs = np.concatenate(songs_splits)
            self.songs_splits["train"].append(train_set_songs)
            self.songs_splits["val"].append(val_set_songs)
            self.songs_splits["test"].append(test_set_songs)

            test_set_labels = labels_splits.pop(cross_val_id)
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
        self, task: str | list[str] = None, cross_val_id: int = 0
    ) -> Dataset:
        self.cross_val_split(cross_val_id=cross_val_id)

        songs = self.songs_splits[self.split]
        labels = self.labels_splits[self.split]

        if task is not None and task != "all":
            if isinstance(task, str):
                songs = songs[labels == self.genres_to_index[task]]
                labels = labels[labels == self.genres_to_index[task]]
            elif isinstance(task, list):
                task = [self.genres_to_index[genre] for genre in task]
                songs = songs[np.isin(labels, task)]
                labels = labels[np.isin(labels, task)]

        return MertGenreClassificationDataset(
            songs=songs,
            labels=labels,
            is_eval=self.is_eval,
            audio_length=self.chunk_length,
            input_sample_rate=self.sample_rate,
        )

    def get_dataloader(
        self,
        task: str = None,
        cross_val_id: int = 0,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> DataLoader:
        dataset = self.get_dataset(task=task, cross_val_id=cross_val_id)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True if (self.split == "train") else False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader
