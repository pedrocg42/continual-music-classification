import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils import data

import config

np.random.seed(config.seed)

GTZAN_GENRES = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9,
}


class GTZANDataset(data.Dataset):
    def __init__(
        self,
        split: str,
        hop_length: int,
        length_spectrogram: int,  # number of spec points in each item
        augment: bool = False,
        split_config: dict = {"train": 0.7, "val": 0.1, "test": 0.2},
        **kwargs
    ):
        self.name = "GTZAN"
        self.dataset_path = config.dataset_path
        self.sr = 22050
        self.genres = GTZAN_GENRES

        self.split = split
        self.split_config = split_config
        self.hop_length = hop_length
        self.length_spectrogram = length_spectrogram
        self.chunk_lengh = self.hop_length * self.length_spectrogram - 1
        self.augment = augment

        self._get_songs()

    def _get_songs(self):
        gtzn_annotations_path = os.path.join(self.dataset_path, "features_30_sec.csv")
        df = pd.read_csv(gtzn_annotations_path)
        song_list = df["filename"].to_numpy()
        song_labels = df["label"].to_numpy()

        idx = np.arange(len(song_list))
        np.random.shuffle(idx)

        train_val_cut = int(len(idx) * self.split_config["train"])
        val_test_cut = int(
            len(idx) * (self.split_config["train"] + self.split_config["val"])
        )

        idx_train = idx[:train_val_cut]
        idx_val = idx[train_val_cut:val_test_cut]
        idx_test = idx[val_test_cut:]

        self.songs = {
            "train": dict(songs=song_list[idx_train], labels=song_labels[idx_train]),
            "val": dict(songs=song_list[idx_val], labels=song_labels[idx_val]),
            "test": dict(songs=song_list[idx_test], labels=song_labels[idx_test]),
        }

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

    def __getitem__(self, index):
        # Get info
        song = self.songs[self.split]["songs"][index]
        label = self.songs[self.split]["labels"][index]

        # Get audio
        audio_filename_path = os.path.join(
            self.dataset_path, "genres", song.split(".")[0], song
        )
        try:
            wav, _ = torchaudio.load(audio_filename_path)
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

    def get_dataloader(self, batch_size: int = 32, num_workers: int = 0):
        data_loader = data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=True if (self.split == "train") else False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader


if __name__ == "__main__":
    dataset = GTZANDataset(split="train", hop_length=512, length_spectrogram=128)

    for item in dataset:
        print("another item")
