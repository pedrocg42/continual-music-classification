# Losses
from torch.nn import CrossEntropyLoss

# Optimizers
from torch.optim import AdamW

# Metrics
from torchmetrics import F1Score, Precision, Recall

# Datasets
from datasets import GTZANDataset, SimpleMusicPipeline

# Experiment Trakcer
from experiment_tracker import TensorboardExperimentTracker

# Architecture
from models import TimmMobileNetV3, TimmMobileViTV2

gtzan_mobilenetv2 = {
    "experiment_name": "gtzan_mobilenetv2",
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
    # experiment tracker
    "experiment_tracker_class": TensorboardExperimentTracker,
}
