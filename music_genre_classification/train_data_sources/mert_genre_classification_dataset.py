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
        song_length: float,
        audio_length: float,
        input_sample_rate: int,
        **kwargs
    ):
        self.songs = songs
        self.labels = labels
        self.is_eval = is_eval
        self.song_length = song_length
        self.audio_length = audio_length
        self.input_sample_rate = input_sample_rate
        self.chunk_lengh = int(self.audio_length * self.input_sample_rate)

    def get_all_chunks_from_song(self, wav: torch.Tensor, label: int):
        # Get number of chunks
        num_chunks = int(np.ceil(wav.shape[0] / self.chunk_lengh))
        # Fill with zeros to process whole song
        temp_wav = wav
        wav = torch.zeros(self.chunk_lengh * num_chunks)
        wav[: len(temp_wav)] = temp_wav
        # Get chunks
        chunks = wav[: num_chunks * self.chunk_lengh].reshape(-1, self.chunk_lengh)
        # Get labels
        labels = torch.tensor([label] * num_chunks).long()
        return chunks, labels

    def get_random_chunk_from_song(self, wav: torch.Tensor, label: int):
        # Get random initial index safely
        if wav.shape[0] >= self.chunk_lengh:
            initial_index = np.random.randint(0, wav.shape[0] - self.chunk_lengh)
        else:
            temp_wav = wav
            wav = torch.zeros(self.chunk_lengh)
            wav[: len(temp_wav)] = temp_wav
            initial_index = 0
        # Get chunk
        chunk = wav[initial_index : initial_index + self.chunk_lengh]
        return chunk, torch.tensor(label).long()

    def __getitem__(self, index):
        if self.is_eval and index >= len(self.songs):
            raise StopIteration()

        index = index % len(self.songs)

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
            return self.get_all_chunks_from_song(wav, label)
        else:
            return self.get_random_chunk_from_song(wav, label)

    def __len__(self):
        if self.is_eval:
            return len(self.songs)
        return len(self.songs) * int(self.song_length / self.audio_length)
