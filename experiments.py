# Metrics
from torchmetrics import F1Score, Precision, Recall

# Criterias
from criterias import TorchCrossEntropyCriteria

# Datasets
from datasets import GTZANDataset

# Experiment Trakcer
from experiment_tracker import TensorboardExperimentTracker

# Looper
from loopers import MusicGenderClassificationLooper

# Model Savers
from model_savers import MusicGenderClassificationModelSaver

# Architecture
from models import TimmMobileNetV3, TimmMobileViTV2

# Optmizers
from optimizers import TorchAdamWOptimizer

# Data Transforms
from train_data_transforms import SimpleMusicPipeline

# Trainers
from trainers import Trainer

gtzan_mobilenetv2_naive = {
    "experiment_name": "gtzan_mobilenetv2_naive",
    # data
    "train": {
        "trainer": Trainer(
            num_epochs=100,
            looper=MusicGenderClassificationLooper(
                train_data_source=GTZANDataset(
                    split="train",
                    hop_length=512,
                    length_spectrogram=128,
                ).get_dataloader(batch_size=128, num_workers=2),
                val_data_source=GTZANDataset(
                    split="val", hop_length=512, length_spectrogram=128
                ),
                train_data_transform=SimpleMusicPipeline(
                    sample_rate=20050,
                    n_fft=1024,
                    win_length=None,
                    hop_length=512,
                    n_mels=128,
                ),
                val_data_transform=SimpleMusicPipeline(
                    sample_rate=20050,
                    n_fft=1024,
                    win_length=None,
                    hop_length=512,
                    n_mels=128,
                ),
                train_model=TimmMobileNetV3(num_classes=10, pretrained=True),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(task="multiclass", num_classes=10),
                        "Precision": Precision(
                            task="multiclass", average="macro", num_classes=10
                        ),
                        "Recall": Recall(
                            task="multiclass", average="macro", num_classes=10
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(task="multiclass", num_classes=10),
                        "Precision": Precision(
                            task="multiclass", average="macro", num_classes=10
                        ),
                        "Recall": Recall(
                            task="multiclass", average="macro", num_classes=10
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {},
}
