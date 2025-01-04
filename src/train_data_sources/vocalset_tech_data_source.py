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


TECHS = [
    "vibrato",
    "straight",
    "belt",
    "breathy",
    "lip_trill",
    "spoken",
    "inhaled",
    "trill",
    "trillo",
    "vocal_fry",
]

SPLIT_SINGERS_CONFIG = {
    "train": [
        "female1",
        "female3",
        "female5",
        "female6",
        "female7",
        "female9",
        "male1",
        "male2",
        "male4",
        "male6",
        "male7",
        "male9",
        "male11",
    ],
    "test": [
        "female2",
        "female8",
        "male3",
        "male5",
        "male10",
    ],
}


class VocalSetTechDataSource(TrainDataSource):
    def __init__(
        self,
        split: str,
        split_singers_config: dict[str, list[str]] = SPLIT_SINGERS_CONFIG,
        techs: list[str] = TECHS,
        is_eval: bool = False,
        chunk_length: float = 5.0,
        **kwargs,
    ):
        self.name = "VocalSetTech"
        self.dataset_path = os.path.join(config.dataset_path, "VocalSet", "FULL")
        self.techs = techs

        # Split parameters
        self.split = split
        self.split_singers_config = split_singers_config
        self.train_val_split_config = dict(train=0.875, val=0.125)
        self.is_eval = is_eval

        # Audio parameters
        self.sample_rate = 44100
        self.song_length = 6  # Average time per clip, useful for training
        self.chunk_length = chunk_length

        self._get_songs()

    def build_label_encoder_and_decoder(self, tasks: list[list[str]]) -> None:
        if tasks == "all" or tasks[0] == "all":
            ordered_techs = self.techs
        else:
            ordered_techs = np.array(flatten_list(tasks)).reshape(-1)
        self.tech_to_index = {tech: i for i, tech in enumerate(ordered_techs)}
        self.index_to_tech = {i: tech for i, tech in enumerate(ordered_techs)}

    def _get_songs(self):
        # Read annotations
        self.songs = np.array(
            glob(os.path.join(self.dataset_path, "*", "*", "*", "*.wav"))
        )

        # Only using audios of the selected techniques
        self.labels = np.array(
            [os.path.normpath(song).split(os.sep)[-2] for song in self.songs]
        )
        mask = np.isin(self.labels, self.techs)
        self.songs = self.songs[mask]
        self.labels = np.array(self.labels[mask])

        # Shuffling
        idx = np.arange(len(self.songs))
        np.random.shuffle(idx)
        self.songs = self.songs[idx]
        self.labels = self.labels[idx]

        # Split
        self.build_splits()

    def build_splits(self):
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

        singers = np.array(
            [os.path.normpath(song).split(os.sep)[-4] for song in self.songs]
        )

        mask_test = np.isin(singers, self.split_singers_config["test"])
        self.songs_splits["test"] = self.songs[mask_test]
        self.labels_splits["test"] = self.labels[mask_test]

        mask_train_val = np.isin(singers, self.split_singers_config["train"])
        train_val_songs = self.songs[mask_train_val]
        train_val_labels = self.labels[mask_train_val]
        for tech in self.techs:
            songs_singer = train_val_songs[train_val_labels == tech]
            labels_singer = train_val_labels[train_val_labels == tech]

            train_idx = round(len(songs_singer) * self.train_val_split_config["train"])

            self.songs_splits["train"].append(songs_singer[:train_idx])
            self.songs_splits["val"].append(songs_singer[train_idx:])
            self.labels_splits["train"].append(labels_singer[:train_idx])
            self.labels_splits["val"].append(labels_singer[train_idx:])

        self.songs_splits["train"] = np.concatenate(self.songs_splits["train"])
        self.songs_splits["val"] = np.concatenate(self.songs_splits["val"])
        self.labels_splits["train"] = np.concatenate(self.labels_splits["train"])
        self.labels_splits["val"] = np.concatenate(self.labels_splits["val"])

    def get_dataset(
        self,
        task: str | list[str] = None,
        tasks: list[list[str]] = ["all"],
        memory_dataset: Dataset = None,
        is_eval: bool | None = None,
    ) -> Dataset:
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
        labels = np.array([self.tech_to_index[tech] for tech in labels])

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
        tasks: list[list[str]] = ["all"],
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
