import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor


class MertGenreClassificationDataset(Dataset):
    def __init__(
        self,
        songs: list,
        labels: list,
        audio_length: float,
        input_sample_rate: int,
        **kwargs
    ):
        self.songs = songs
        self.labels = labels
        self.audio_length = audio_length
        self.input_sample_rate = input_sample_rate
        self.chunk_lengh = int(self.audio_length * self.input_sample_rate)

    def __getitem__(self, index):
        # Get info
        song_path = self.songs[index]
        label = self.labels[index]

        # Get audio
        try:
            output_wav = torch.zeros((self.chunk_lengh))
            wav, _ = torchaudio.load(song_path)
            output_wav[: wav.shape[1]] = wav[0, : self.chunk_lengh]
        except:
            return (
                torch.zeros((1, self.chunk_lengh)),
                torch.tensor(label),
            )

        return output_wav, torch.tensor(label).long()

    def __len__(self):
        return len(self.songs)
