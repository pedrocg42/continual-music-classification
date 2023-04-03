import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class MusicGenreClassificationDataset(Dataset):
    def __init__(
        self, songs: list, labels: list, split: str, chunk_length: int, **kwargs
    ):
        self.songs = songs
        self.labels = labels
        self.split = split
        self.chunk_lengh = chunk_length

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
                torch.tensor(label),
            )

        if self.split == "train":
            return wav, torch.tensor(label).long()
        else:
            return wav, torch.tensor(label).long().repeat(num_chunks)

    def __len__(self):
        return len(self.songs)

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
