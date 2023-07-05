import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class MertGenreClassificationDataset(Dataset):
    def __init__(
        self,
        songs: list,
        labels: list,
        is_eval: bool,
        audio_length: float,
        input_sample_rate: int,
        **kwargs
    ):
        self.songs = songs
        self.labels = labels
        self.is_eval = is_eval
        self.audio_length = audio_length
        self.input_sample_rate = input_sample_rate
        self.chunk_lengh = int(self.audio_length * self.input_sample_rate)

    def get_all_chunks_from_song(self, wav: torch.Tensor, label: int):
        # Get number of chunks
        num_chunks = wav.shape[0] // self.chunk_lengh

        # Get chunks
        chunks = wav[: num_chunks * self.chunk_lengh].reshape(-1, self.chunk_lengh)

        # Get labels
        labels = torch.tensor([label] * num_chunks).long()

        return chunks, labels

    def get_random_chunk_from_song(self, wav: torch.Tensor):
        # Get random initial index
        initial_index = np.random.randint(0, wav.shape[0] - self.chunk_lengh)

        # Get chunk
        chunk = wav[initial_index : initial_index + self.chunk_lengh]

        return chunk

    def __getitem__(self, index):
        # Get info
        song_path = self.songs[index]
        label = self.labels[index]

        # Get audio
        try:
            wav, _ = torchaudio.load(song_path)
        except:
            wav = torch.zeros((1, 10 * self.chunk_lengh))

        # Removing channel dimension
        wav = wav.squeeze(0)

        if self.is_eval:
            output_wav, labels = self.get_all_chunks_from_song(wav, label)
            return output_wav, labels
        else:
            output_wav = self.get_random_chunk_from_song(wav)
            return output_wav, torch.tensor(label).long()

    def __len__(self):
        return len(self.songs)
