# Metrics
from torchmetrics import F1Score, Precision, Recall

# Criterias
from criterias import TorchCrossEntropyCriteria

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

# Datasets
from train_data_sources import GtzanDataSource

# Data Transforms
from train_data_transforms import SimpleMusicPipeline

# Trainers
from trainers import Trainer, ContinualLearningTrainer

# Evaluator
from evaluators import TasksEvaluator, ContinualLearningTasksEvaluator

###############################################################
###########           GENERIC COMPONENTS            ###########
###############################################################

data_transform = SimpleMusicPipeline(
    sample_rate=20050,
    n_fft=1024,
    win_length=None,
    hop_length=512,
    n_mels=128,
)

###############################################################
###########                BASELINES                ###########
###############################################################

gtzan_mobilenetv2_joint = {
    "experiment_name": "gtzan_mobilenetv2_joint",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": 5,
    # data
    "train": {
        "trainer": Trainer(
            num_epochs=200,
            early_stopping_patience=40,
            early_stopping_metric="F1 Score",
            looper=MusicGenderClassificationLooper(
                train_data_source=GtzanDataSource(
                    split="train",
                    num_cross_val_splits=5,
                    hop_length=512,
                    length_spectrogram=128,
                ),
                val_data_source=GtzanDataSource(
                    split="val",
                    num_cross_val_splits=5,
                    hop_length=512,
                    length_spectrogram=128,
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
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=10
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=10
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=10
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=10
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=10
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=10
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": TasksEvaluator(
            model=TorchClassificationModel(
                encoder_name="dino_resnet50", pretrained=True, num_classes=10
            ),
            model_saver=MusicGenderClassificationModelSaver(),
            data_source=GtzanDataSource(
                split="test",
                num_cross_val_splits=5,
                hop_length=512,
                length_spectrogram=128,
            ),
            data_transform=data_transform,
            metrics={
                "F1 Score": F1Score(task="multiclass", average="none", num_classes=10),
                "Precision": Precision(
                    task="multiclass", average="none", num_classes=10
                ),
                "Recall": Recall(task="multiclass", average="none", num_classes=10),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}


###############################################################
###########                CONTINUAL                ###########
###############################################################
scenario1 = [
    [
        "blues",
        "classical",
    ],
    [
        "country",
        "disco",
    ],
    [
        "hiphop",
        "jazz",
    ],
    [
        "metal",
        "pop",
    ],
    [
        "reggae",
        "rock",
    ],
]

gtzan_mobilenetv2_scenario1 = {
    "experiment_name": "gtzan_mobilenetv2_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Naive",
    "num_cross_val_splits": 5,
    # data
    "train": {
        "trainer": ContinualLearningTrainer(
            tasks=scenario1,
            num_epochs=200,
            early_stopping_patience=40,
            early_stopping_metric="F1 Score",
            looper=MusicGenderClassificationLooper(
                train_data_source=GtzanDataSource(
                    split="train",
                    num_cross_val_splits=5,
                    hop_length=512,
                    length_spectrogram=128,
                ),
                val_data_source=GtzanDataSource(
                    split="val",
                    num_cross_val_splits=5,
                    hop_length=512,
                    length_spectrogram=128,
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
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=10
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=10
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=10
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=10
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=10
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=10
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluator(
            tasks=scenario1,
            model=TorchClassificationModel(
                encoder_name="dino_resnet50", pretrained=True, num_classes=10
            ),
            model_saver=MusicGenderClassificationModelSaver(),
            data_source=GtzanDataSource(
                split="test",
                num_cross_val_splits=5,
                hop_length=512,
                length_spectrogram=128,
            ),
            data_transform=data_transform,
            metrics={
                "F1 Score": F1Score(task="multiclass", average="none", num_classes=10),
                "Precision": Precision(
                    task="multiclass", average="none", num_classes=10
                ),
                "Recall": Recall(task="multiclass", average="none", num_classes=10),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}
