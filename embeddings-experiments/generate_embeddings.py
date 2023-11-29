import numpy as np
import torch
from fire import Fire

from music_genre_classification.models.embedding_model import TorchEmbeddingModel
from music_genre_classification.train_data_sources.train_data_source_factory import (
    TrainDataSourceFactory,
)
from music_genre_classification.train_data_transforms.mert_data_transform import (
    MertDataTransform,
)
import config
from loguru import logger
from tqdm import tqdm

torch.set_grad_enabled(False)


def generate_embeddings(
    encoder_config: dict = dict(name="MertEncoder"),
    # data_config: dict = dict(name="GtzanDataSource"),
    # input_sample_rate: int = 22050, # GTZAN
    # data_config: dict = dict(name="VocalSetSingerDataSource"),
    # data_config: dict = dict(name="VocalSetTechDataSource"),
    # input_sample_rate: int = 44100,  # VocalSet
    data_config: dict = dict(
        name="NSynthInstrumentTechDataSource", args=dict(num_items_per_class=-1)
    ),
    input_sample_rate: int = 16000,  # NSynth
    output_name: str = "nsynth",
):
    embedding_model = TorchEmbeddingModel(encoder=encoder_config).to(config.device)
    data_transform = MertDataTransform(input_sample_rate=input_sample_rate).to(
        config.device
    )

    for split in ["train", "val", "test"]:
        logger.info(f"Starting embedding extraction of split: {split}")

        data_config["args"].update(dict(split=split, is_eval=True))
        data_source = TrainDataSourceFactory().build(config=data_config)
        dataset = data_source.get_dataset()

        embeddings = []
        labels = []
        for chunks, chunks_labels in tqdm(dataset, desc="Embedding Extraction"):
            if chunks is None:
                continue
            chunks = chunks.to(config.device, non_blocking=True)

            chunks_transformed = data_transform(chunks)
            chunks_embeddings = embedding_model.forward_features(chunks_transformed)

            embeddings.append(chunks_embeddings.cpu().numpy())
            labels += [
                # data_source.index_to_genre[index] for index in chunks_labels.numpy()
                data_source.index_to_instrument[index]
                for index in chunks_labels.numpy()
            ]

        embeddings = np.concatenate(embeddings)
        labels = np.array(labels)

        np.save(f"{output_name}_{split}_embeddings.npy", embeddings)
        np.save(f"{output_name}_{split}_labels.npy", labels)

    logger.info("Finished embeddings extraction")


if __name__ == "__main__":
    Fire(generate_embeddings)
