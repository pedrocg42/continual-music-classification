# Found filtered dataset from https://github.com/coreyker/dnn-mgr/tree/master/gtzan
import os
from glob import glob

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import config
from src.my_utils.lists import flatten_list
from src.train_data_sources.mert_genre_classification_dataset import (
    MertGenreClassificationDataset,
)
from src.train_data_sources.train_data_source import (
    TrainDataSource,
)

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
        self,
        split: str,
        is_eval: bool = False,
        chunk_length: float = 5.0,
        **kwargs,
    ):
        self.name = "GTZAN"
        self.dataset_path = os.path.join(config.dataset_path, self.name)
        self.genres = GTZAN_GENRES

        # Split parameters
        self.split = split
        self.is_eval = is_eval

        # Audio parameters
        self.sample_rate = 22050
        self.song_length = 30
        self.chunk_length = chunk_length

        self._get_songs()

    def build_label_encoder_and_decoder(self, tasks: list[list[str]]) -> None:
        if tasks == "all" or tasks[0] == "all":
            ordered_genres = self.genres
        else:
            ordered_genres = np.array(flatten_list(tasks)).reshape(-1)
        self.genres_to_index = {genre: i for i, genre in enumerate(ordered_genres)}
        self.index_to_genre = {i: genre for i, genre in enumerate(ordered_genres)}

    def _get_songs(self):
        # Split
        self.songs_splits = {}
        self.labels_splits = {}
        # Read annotations
        song_list = np.array(glob(os.path.join(self.dataset_path, "genres", "*", "*.wav")))
        song_labels = np.array([os.path.basename(song).split(".")[0] for song in song_list])

        # Filtered songs
        list_accepted_songs = []
        for split_filename, split in [
            ("train", "train"),
            ("valid", "val"),
            ("test", "test"),
        ]:
            with open(os.path.join(self.dataset_path, f"{split_filename}_filtered.txt")) as f:
                list_accepted_songs = f.readlines()
            list_accepted_songs = np.array([os.path.basename(song).split(".wav")[0] for song in list_accepted_songs])
            mask = np.array([os.path.basename(song).split(".wav")[0] in list_accepted_songs for song in song_list])
            self.songs_splits[split] = song_list[mask]
            self.labels_splits[split] = song_labels[mask]

    def get_dataset(
        self,
        task: str | list[str] = None,
        tasks: list[list[str]] = None,
        memory_dataset: Dataset = None,
        is_eval: bool | None = None,
    ) -> Dataset:
        if tasks is None:
            tasks = ["all"]
        self.build_label_encoder_and_decoder(tasks)

        songs = self.songs_splits[self.split]
        labels = self.labels_splits[self.split]

        if task is not None and task != "all":
            if isinstance(task, str):
                songs = songs[labels == task]
                labels = labels[labels == task]
            elif isinstance(task, list):
                songs = songs[np.isin(labels, task)]
                labels = labels[np.isin(labels, task)]
        labels = np.array([self.genres_to_index[genre] for genre in labels])

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
        tasks: list[list[str]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        if tasks is None:
            tasks = ["all"]
        dataset = self.get_dataset(task=task, tasks=tasks, **kwargs)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(self.split == "train"),
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader
