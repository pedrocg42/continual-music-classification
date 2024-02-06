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


def generate_embeddings():

    dataset_configs = {
        "gtzan": {
            "data_config": dict(name="GtzanDataSource"),
            "input_sample_rate": 22050,
        },
        "vocalset-singer": {
            "data_config": dict(name="VocalSetSingerDataSource"),
            "input_sample_rate": 44100,
        },
        "vocalset-tech": {
            "data_config": dict(name="VocalSetTechDataSource"),
            "input_sample_rate": 44100,
        },
        "nsynth": {
            "data_config": dict(name="NSynthInstrumentTechDataSource"),
            "input_sample_rate": 16000,
        },
    }

    for output_name, kwargs in dataset_configs.items():
        logger.info(f"Starting embedding extraction of dataset: {output_name}")
        generate_dataset_embeddings(output_name=output_name, **kwargs)


def generate_dataset_embeddings(
    encoder_config: dict = dict(name="MertEncoder"),
    data_config: dict = dict(name="GtzanDataSource"),
    input_sample_rate: int = 22050,
    output_name: str = "gtzan",
):
    embedding_model = (
        TorchEmbeddingModel(encoder=encoder_config).to(config.device).eval()
    )
    data_transform = MertDataTransform(input_sample_rate=input_sample_rate).to(
        config.device
    )

    for split in ["train", "val", "test"]:
        logger.info(f"Starting embedding extraction of split: {split}")

        if output_name == "nsynth" and split == "train":
            data_config["args"] = dict(split=split, is_eval=True, num_items_per_class=5000)
        else:
            data_config["args"] = dict(split=split, is_eval=True)

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

            embeddings.append(
                chunks_embeddings.cpu().numpy().mean(axis=0, keepdims=True)
            )
            labels.append(chunks_labels[0].numpy())

        embeddings = np.concatenate(embeddings)
        labels = np.array(labels)

        np.save(f"{output_name}_{split}_embeddings.npy", embeddings)
        np.save(f"{output_name}_{split}_labels.npy", labels)

    logger.info("Finished embeddings extraction")


if __name__ == "__main__":
    Fire(generate_embeddings)
