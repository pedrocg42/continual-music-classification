# Metrics
from torchmetrics import F1Score, Precision, Recall

# Criterias
from criterias import TorchCrossEntropyCriteria

# Datasets
from train_data_sources import GTZANDataset

# Evaluator
from evaluators import Evaluator

# Experiment Trakcer
from experiment_tracker import DataframeExperimentTracker, TensorboardExperimentTracker

# Looper
from loopers import MusicGenderClassificationLooper

# Model Savers
from model_savers import MusicGenderClassificationModelSaver

# Architecture
from models import TorchClassificationModel

# Optmizers
from optimizers import TorchAdamWOptimizer

# Data Transforms
from train_data_transforms import SimpleMusicPipeline

# Trainers
from trainers import Trainer

data_transform = SimpleMusicPipeline(
    sample_rate=20050,
    n_fft=1024,
    win_length=None,
    hop_length=512,
    n_mels=128,
)
gtzan_mobilenetv2_joint = {
    "experiment_name": "gtzan_mobilenetv2_cumulative",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    # data
    "train": {
        "trainer": Trainer(
            num_epochs=200,
            early_stopping_patience=10,
            early_stopping_metric="F1 Score",
            looper=MusicGenderClassificationLooper(
                train_data_source=GTZANDataset(
                    split="train", hop_length=512, length_spectrogram=128
                ).get_dataloader(batch_size=32, num_workers=0),
                val_data_source=GTZANDataset(
                    split="val", hop_length=512, length_spectrogram=128
                ),
                train_data_transform=data_transform,
                val_data_transform=data_transform,
                train_model=TorchClassificationModel(
                    encoder_name="dino_resnet50", pretrained=True, num_classes=10
                ),
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
    "evaluate": {
        "evaluator": Evaluator(
            model=TorchClassificationModel(
                encoder_name="dino_resnet50", pretrained=True, num_classes=10
            ),
            model_saver=MusicGenderClassificationModelSaver(),
            data_source=GTZANDataset(
                split="val", hop_length=512, length_spectrogram=128
            ),
            data_transform=data_transform,
            metrics={
                "F1 Score": F1Score(task="multiclass", num_classes=10),
                "Precision": Precision(
                    task="multiclass", average="macro", num_classes=10
                ),
                "Recall": Recall(task="multiclass", average="macro", num_classes=10),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}
