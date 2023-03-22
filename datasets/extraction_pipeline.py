import torch
from torchaudio.transforms import (
    FrequencyMasking,
    MelScale,
    MelSpectrogram,
    Resample,
    Spectrogram,
    TimeMasking,
    TimeStretch,
)


class SimpleMusicPipeline(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        raw_audio_augmentation: torch.nn.Module = None,
        spectrogram_augmentation: torch.nn.Module = None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.raw_audio_augmentation = raw_audio_augmentation
        self.spectrogram_augmentation = spectrogram_augmentation

        self.mel_transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

    def forward(self, waveform: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if self.raw_audio_augmentation is not None and augment:
            waveform = self.raw_audio_augmentation(waveform)

        spec = self.mel_transform(waveform)

        if self.raw_audio_augmentation is not None and augment:
            spec = self.spectrogram_augmentation(spec)

        return spec


# Define custom feature extraction pipeline.
#
# 1. Resample audio
# 2. Convert to power spectrogram
# 3. Apply augmentations
# 4. Convert to mel-scale
#
class AudioResamplePipeline(torch.nn.Module):
    def __init__(
        self,
        input_freq=16000,
        resample_freq=8000,
        n_fft=1024,
        n_mel=256,
        stretch_factor=0.8,
    ):
        super().__init__()
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

        self.spec = Spectrogram(n_fft=n_fft, power=2)

        self.spec_aug = torch.nn.Sequential(
            TimeStretch(stretch_factor, fixed_rate=True),
            FrequencyMasking(freq_mask_param=80),
            TimeMasking(time_mask_param=80),
        )

        self.mel_scale = MelScale(
            n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        resampled = self.resample(waveform)

        # Convert to power spectrogram
        spec = self.spec(resampled)

        # Apply SpecAugment
        spec = self.spec_aug(spec)

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        return mel
