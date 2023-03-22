# Datasets
from datasets import GTZANDataset

# Datasets
from datasets import GTZANDataset
from datasets import SimpleMusicPipeline

# Architecture
from models import TimmMobileNetV3
from models import TimmMobileViTV2

# Optimizers
from torch.optim import AdamW

# Losses
from torch.nn import CrossEntropyLoss

# Metrics
from torchmetrics import F1Score, Precision, Recall


gtzan_mobilenetv2 = {
    "name": "gtzan_mobilenetv2",
    # data
    "dataset_class": GTZANDataset,
    "sample_rate": 20050,
    "hop_length": 512,
    "length_spectrogram": 128,
    # transforms
    "extraction_pipeline": SimpleMusicPipeline(
        sample_rate=20050,
        n_fft=1024,
        win_length=None,
        hop_length=512,
        n_mels=128,
    ),
    # architecture
    "architecture": TimmMobileNetV3,
    "num_classes": 10,
    "pretrained": True,
    # train
    "criteria_class": CrossEntropyLoss,
    "optimizer_class": AdamW,
    "learning_rate": 3e-4,
    "num_epochs": 100,
    "batch_size": 32,
    # metrics
    "metrics": {
        "train": {
            "F1 Score": F1Score(task="multiclass", num_classes=10),
            "Precision": Precision(task="multiclass", average="macro", num_classes=10),
            "Recall": Recall(task="multiclass", average="macro", num_classes=10),
        },
        "val": {
            "F1 Score": F1Score(task="multiclass", num_classes=10),
            "Precision": Precision(task="multiclass", average="macro", num_classes=10),
            "Recall": Recall(task="multiclass", average="macro", num_classes=10),
        },
    },
}
