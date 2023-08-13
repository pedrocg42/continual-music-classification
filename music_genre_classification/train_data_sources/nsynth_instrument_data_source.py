# Found filtered dataset from https://github.com/coreyker/dnn-mgr/tree/master/gtzan
import os
from glob import glob

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import config
from music_genre_classification.my_utils.lists import flatten_list
from music_genre_classification.train_data_sources.mert_genre_classification_dataset import (
    MertGenreClassificationDataset,
)
from music_genre_classification.train_data_sources.train_data_source import (
    TrainDataSource,
)

np.random.seed(config.seed)


INSTRUMENTS = [
    "bass",
    "brass",
    "flute",
    "guitar",
    "keyboard",
    "mallet",
    "organ",
    "reed",
    "string",
    "synth_lead",
    "vocal",
]


class NSynthInstrumentTechDataSource(TrainDataSource):
    def __init__(
        self,
        split: str,
        instruments: list[str] = INSTRUMENTS,
        is_eval: bool = False,
        chunk_length: float = 5.0,
        **kwargs,
    ):
        self.name = "NSynthInstrument"
        self.dataset_path = os.path.join(config.dataset_path, "NSynth")
        self.instruments = instruments

        # Split parameters
        self.split = split
        self.is_eval = is_eval

        # Audio parameters
        self.sample_rate = 44100
        self.song_length = 6  # Average time per clip, useful for training
        self.chunk_length = chunk_length

        self._get_songs()

    def build_label_encoder_and_decoder(self, tasks: list[list[str]]) -> None:
        if tasks == "all" or tasks[0] == "all":
            ordered_instruments = self.instruments
        else:
            ordered_instruments = np.array(flatten_list(tasks)).reshape(-1)
        self.instrument_to_index = {
            instrument: i for i, instrument in enumerate(ordered_instruments)
        }
        self.index_to_instrument = {
            i: instrument for i, instrument in enumerate(ordered_instruments)
        }

    def _get_songs(self):
        self.songs_splits = {}
        self.labels_splits = {}

        for folder_split_name, split in [
            ("nsynth-train", "train"),
            ("nsynth-valid", "val"),
            ("nsynth-test", "test"),
        ]:
            # Read annotations
            self.songs_splits[split] = np.array(
                glob(
                    os.path.join(self.dataset_path, folder_split_name, "audio", "*.wav")
                )
            )

            song_labels = []
            for song in self.songs_splits[split]:
                splitted_name = os.path.basename(song).split("_")
                match len(splitted_name):
                    case 3:
                        label = splitted_name[0]
                    case 4:
                        label = "_".join(splitted_name[:2])
                    case _:
                        raise NotImplementedError()
                song_labels.append(label)
            self.labels_splits[split] = np.array(song_labels)

    def get_dataset(
        self,
        task: str | list[str] = None,
        tasks: list[list[str]] = 0,
        memory_dataset: Dataset = None,
        is_eval: bool | None = None,
    ) -> Dataset:
        self.build_label_encoder_and_decoder(tasks)

        songs = self.songs_splits[self.split]
        labels = np.array(
            [
                self.instrument_to_index[instrument]
                for instrument in self.labels_splits[self.split]
            ]
        )

        if task is not None and task != "all":
            if isinstance(task, str):
                songs = songs[labels == self.instrument_to_index[task]]
                labels = labels[labels == self.instrument_to_index[task]]
            elif isinstance(task, list):
                task = [self.instrument_to_index[instrument] for instrument in task]
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
