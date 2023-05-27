import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class MusicGenreClassificationDataset(Dataset):
    def __init__(
        self,
        songs: list,
        labels: list,
        split: str,
        sample_rate: int,
        song_length: float,
        **kwargs
    ):
        self.songs = songs
        self.labels = labels
        self.split = split
        self.sample_rate = sample_rate
        self.song_length = song_length
        self.chunk_lengh = int(self.sample_rate * self.song_length)

    def __getitem__(self, index):
        # Get info
        song_path = self.songs[index]
        label = self.labels[index]

        # Get audio
        try:
            output_wav = torch.zeros((1, self.chunk_lengh))
            wav, _ = torchaudio.load(song_path)
            output_wav[:, : wav.shape[1]] = wav[:, : self.chunk_lengh]
        except:
            return (
                torch.zeros((1, self.chunk_lengh)),
                torch.tensor(label),
            )

        return output_wav, torch.tensor(label).long()

    def __len__(self):
        return len(self.songs)
