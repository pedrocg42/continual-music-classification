import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

import config
from music_genre_classification.loopers.music_genre_classification_looper import (
    MusicGenreClassificationLooper,
)


class DkvbMusicGenreClassificationLooper(MusicGenreClassificationLooper):
    @torch.no_grad()
    def key_init_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        data_loader: DataLoader,
        data_transform: torch.nn.Module,
    ):
        logger.info(f"Key initialization epoch {epoch + 1}")
        model.prepare_keys_initialization()
        pbar = tqdm(
            data_loader,
            colour="#5ee0f7",
            total=self.max_steps if self.debug else len(data_loader),
        )
        for i, (waveforms, _) in enumerate(pbar):
            if self.debug and i == self.max_steps:
                break
            self.key_init_batch(waveforms, model, data_transform)

    @torch.no_grad()
    def key_init_batch(
        self,
        waveforms: torch.Tensor,
        model: torch.nn.Module,
        data_transform: torch.nn.Module,
    ):
        waveforms = waveforms.to(config.device)

        # Inference
        transformed = data_transform(waveforms)
        model(transformed)
