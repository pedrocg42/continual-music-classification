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
        num_items_per_class: int = -1,
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
        self.num_items_per_class = num_items_per_class

        # Audio parameters
        self.sample_rate = 16000
        self.song_length = 4  # Average time per clip, useful for training
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

            if split == "train" and self.num_items_per_class > 0:
                instrument_subtypes = []
                for song in self.songs_splits[split]:
                    splitted_name = os.path.basename(song).split("_")
                    match len(splitted_name):
                        case 3:
                            label = splitted_name[1]
                        case 4:
                            label = splitted_name[2]
                        case _:
                            raise NotImplementedError()
                    instrument_subtypes.append(label)
                instrument_subtypes = np.array(instrument_subtypes)

                song_paths = []
                song_labels = []
                for instrument in self.instruments:
                    mask = self.labels_splits[split] == instrument
                    temp_paths = self.songs_splits[split][mask]
                    temp_labels = self.labels_splits[split][mask]
                    temp_subtypes = instrument_subtypes[mask]
                    already_included_samples = 0
                    for i in range(101):
                        unique_subtypes, unique_subtypes_counts = np.unique(
                            temp_subtypes, return_counts=True
                        )
                        num_items_per_subtype = (
                            self.num_items_per_class - already_included_samples
                        ) // len(unique_subtypes)
                        if all(unique_subtypes_counts > num_items_per_subtype):
                            for subtype in unique_subtypes:
                                mask = np.where(temp_subtypes == subtype)[0]
                                np.random.shuffle(mask)
                                song_paths.append(
                                    temp_paths[mask[:num_items_per_subtype]]
                                )
                                song_labels.append(
                                    temp_labels[mask[:num_items_per_subtype]]
                                )
                            break
                        else:
                            # Including all items from subtypes with not enough samples
                            for subtype in unique_subtypes[
                                unique_subtypes_counts < num_items_per_subtype
                            ]:
                                mask = temp_subtypes == subtype
                                already_included_samples += np.count_nonzero(mask)
                                song_paths.append(temp_paths[mask])
                                song_labels.append(temp_labels[mask])
                                # Removing samples from already included class
                                temp_paths = temp_paths[~mask]
                                temp_labels = temp_labels[~mask]
                                temp_subtypes = temp_subtypes[~mask]
                self.songs_splits[split] = np.concatenate(song_paths)
                self.labels_splits[split] = np.concatenate(song_labels)

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
        labels = np.array(
            [self.instrument_to_index[instrument] for instrument in labels]
        )

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
