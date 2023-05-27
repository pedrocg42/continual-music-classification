import torch
import torchaudio.transforms as T
from music_genre_classification.train_data_transforms.train_data_transform import (
    TrainDataTransform,
)
from transformers import Wav2Vec2FeatureExtractor
import config

class MertDataTransform(TrainDataTransform):
    def __init__(
        self,
        input_sample_rate: int,
        output_sample_rate: int = 24000,
    ):
        super().__init__()

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.resampler = T.Resample(input_sample_rate, output_sample_rate)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )

    def transform(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        inputs = self.resampler(inputs)
        inputs = {"input_values": inputs, "attention_mask": torch.ones(len(inputs), device=config.device)}
        return inputs

    def to(self, device: torch.device, **kwargs) -> None:
        self.resampler.to(device)
