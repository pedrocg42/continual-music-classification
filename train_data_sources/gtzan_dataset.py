import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from train_data_sources.train_data_source import TrainDataSource

import config

np.random.seed(config.seed)


class MusicGenreClassificationDataset(Dataset):
    def __init__(self, songs: list, labels: list, split: str, **kwargs):
        self.songs = songs
        self.labels = labels
        self.split = split

    def __getitem__(self, index):
        # Get info
        song_path = self.songs[index]
        label = self.labels[index]

        # Get audio
        try:
            wav, _ = torchaudio.load(song_path)
            wav, num_chunks = self._adjust_audio_length(wav)
        except:
            return (
                torch.zeros((1, self.chunk_lengh)),
                torch.tensor(self.genres[label]),
            )

        if self.split == "train":
            return wav, torch.tensor(self.genres[label])
        else:
            return wav, torch.tensor(self.genres[label]).repeat(num_chunks)

    def __len__(self):
        return len(self.songs[self.split]["songs"])

    def _adjust_audio_length(self, wav: torch.Tensor) -> torch.Tensor:
        if self.split == "train":
            random_index = np.random.randint(0, wav.shape[-1] - self.chunk_lengh)
            return wav[:, random_index : random_index + self.chunk_lengh], 1
        else:
            num_chunks = wav.shape[-1] // self.chunk_lengh
            return (
                torch.reshape(
                    wav[0, : self.chunk_lengh * num_chunks],
                    (-1, 1, self.chunk_lengh),
                ),
                num_chunks,
            )


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
        self,
        split: str,
        num_cross_val_splits: int = 5,
        hop_length: int = 512,
        length_spectrogram: int = 128,
        **kwargs
    ):
        self.name = "GTZAN"
        self.dataset_path = config.dataset_path
        self.genres = GTZAN_GENRES
        self.genres_to_index = {genre: i for i, genre in enumerate(self.genres)}
        self.index_to_genres = {i: genre for i, genre in enumerate(self.genres)}

        # Split parameters
        self.split = split
        self.num_cross_val_splits = num_cross_val_splits

        # Audio parameters
        self.sr = 22050
        self.hop_length = hop_length
        self.length_spectrogram = length_spectrogram
        self.chunk_lengh = self.hop_length * self.length_spectrogram - 1

        self._get_songs()

    def _get_songs(self):
        # Read annotations
        gtzn_annotations_path = os.path.join(self.dataset_path, "features_30_sec.csv")
        self.df = pd.read_csv(gtzn_annotations_path)
        song_list = self.df["filename"].to_numpy()
        song_labels = self.df["label"].to_numpy()

        # Transform
        self.songs = np.array(
            [
                os.path.join(self.dataset_path, "genres", song.split(".")[0], song)
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

    def get_dataset(self, task: str = "all", cross_val_id: int = 0) -> Dataset:
        self.cross_val_split(cross_val_id=cross_val_id)

        songs = self.songs_splits[self.split]
        labels = self.labels_splits[self.split]

        if task is not "all":
            songs = songs[labels == self.genres_to_index[task]]
            labels = labels[labels == self.genres_to_index[task]]

        return MusicGenreClassificationDataset(
            songs=songs, labels=labels, split=self.split
        )

    def get_dataloader(
        self,
        task: str = "all",
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
