import os

import numpy as np
import timm
import torch
import torch.nn as nn
from datasets import load_dataset
from fire import Fire
from loguru import logger
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.transforms.functional import normalize, pil_to_tensor, resize
from tqdm import tqdm

torch.set_grad_enabled(False)

model_name_to_timm = {
    "resnet18": "resnet18.tv_in1k",
    "resnet34": "resnet34.tv_in1k",
    "resnet50": "resnet50.tv_in1k",
    "resnet101": "resnet101.tv_in1k",
    "vit": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "vit-dino": "vit_base_patch16_224.dino",
}


def collate_fn_cifar100(data):
    images = [None] * len(data)
    labels = [None] * len(data)
    for i, item_dict in enumerate(data):
        images[i] = pil_to_tensor(item_dict["img"]).to(torch.float32)[None] / 255.0
        labels[i] = torch.as_tensor(item_dict["fine_label"], dtype=torch.long)[None]
    images = torch.concat(images)
    labels = torch.concat(labels)
    return images, labels


def collate_fn_core50(data):
    images = [None] * len(data)
    labels = [None] * len(data)
    for i, item_dict in enumerate(data):
        images[i] = pil_to_tensor(item_dict["image"]).to(torch.float32)[None] / 255.0
        labels[i] = torch.as_tensor(item_dict["label"], dtype=torch.long)[None]
    images = torch.concat(images)
    labels = torch.concat(labels)
    return images, labels


def build_data_loader(dataset_name: str, split: str) -> torch.utils.data.DataLoader:
    match dataset_name:
        case "cifar100":
            dataset = load_dataset("uoft-cs/cifar100", split=split)
            collate_fn = collate_fn_cifar100
        case "core50":
            dataset = load_dataset("adrake17/core50", split=split)
            collate_fn = collate_fn_core50
        case "google-landmark-v2":
            # manual dataset
            pass
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count() // 2, collate_fn=collate_fn
    )
    data_loader_w_progress = tqdm(data_loader, desc=f"Extracting {split} embeddings")
    return data_loader_w_progress


def generate_embeddings():
    for model_name in ["resnet18", "resnet34", "resnet50", "vit", "vit-dino"]:
        for dataset_name in ["cifar100", "core50"]:
            logger.info(f"Starting embedding extraction for model {model_name} of dataset: {dataset_name}")
            generate_dataset_embeddings(model_name, dataset_name)
            logger.info(f"Finished embedding extraction for model {model_name} of dataset: {dataset_name}")


def generate_dataset_embeddings(
    model_name: str,
    dataset_name: str,
):
    logger.info(f"Loading model: {model_name}")
    model: nn.Module = timm.create_model(model_name_to_timm[model_name], pretrained=True).eval().to("cuda")

    for split in ["train", "test"]:
        output_folder = f"embeddings-experiments/image/{model_name}/{dataset_name}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        output_name = f"{output_folder}/{model_name}_{dataset_name}_{split}"

        data_loader = build_data_loader(dataset_name, split)
        logger.info(f"Starting embedding extraction of split: {split}")

        embeddings = []
        labels = []
        for i, (images, labels_batch) in enumerate(data_loader):
            if model_name == "vit":
                images = resize(normalize(images, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD), (224, 224))
            else:
                images = resize(normalize(images, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD), (224, 224))
            embeddings_batch = model.forward_features(images.to("cuda"))
            embeddings_batch = model.forward_head(embeddings_batch, pre_logits=True)
            embeddings.append(embeddings_batch.detach().cpu().numpy())
            labels.append(labels_batch.numpy())

            if (i + 1) % 100 == 0:
                np.save(f"{output_name}_embeddings.npy", np.concatenate(embeddings))
                np.save(f"{output_name}_labels.npy", np.concatenate(labels))

        np.save(f"{output_name}_embeddings.npy", np.concatenate(embeddings))
        np.save(f"{output_name}_labels.npy", np.concatenate(labels))

    logger.info("Finished embeddings extraction")


if __name__ == "__main__":
    Fire(generate_embeddings)
