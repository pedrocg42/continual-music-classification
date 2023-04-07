import torch
from loguru import logger
from tqdm import tqdm

import config
from loopers.music_gender_classification_looper import MusicGenderClassificationLooper


class DkvbMusicGenderClassificationLooper(MusicGenderClassificationLooper):
    @torch.no_grad()
    def key_init_epoch(self, epoch: int):
        logger.info(f"Key initialization epoch {epoch + 1}")
        self.model.prepare_keys_initialization()
        pbar = tqdm(self.train_data_loader, colour="#5ee0f7")
        for waveforms, _ in pbar:
            self.key_init_batch(waveforms)

    @torch.no_grad()
    def key_init_batch(self, waveforms: torch.Tensor):
        waveforms = waveforms.to(config.device)

        # Inference
        spectrograms = self.val_data_transform(waveforms)
        self.model(spectrograms)
