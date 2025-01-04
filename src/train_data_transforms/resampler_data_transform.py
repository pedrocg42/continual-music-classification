import torch
import torchaudio.transforms as T

import config
from src.train_data_transforms.train_data_transform import (
    TrainDataTransform,
)


class ResamplerDataTransform(TrainDataTransform):
    def __init__(
        self,
        input_sample_rate: int,
        output_sample_rate: int = 24000,
    ):
        super().__init__()

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.resampler = T.Resample(input_sample_rate, output_sample_rate)

    def transform(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        if inputs.dim() == 2:
            inputs = inputs[:, None, :]
        inputs = self.resampler(inputs)
        return inputs

    def to(self, device: torch.device, **kwargs) -> None:
        self.resampler.to(device)
        return self
