# Metrics
from torchmetrics import F1Score, Precision, Recall

# Criterias
from criterias import TorchCrossEntropyCriteria

# Experiment Trakcer
from experiment_tracker import DataframeExperimentTracker, TensorboardExperimentTracker

# Looper
from loopers import MusicGenderClassificationLooper, DkvbMusicGenderClassificationLooper

# Model Savers
from model_savers import MusicGenderClassificationModelSaver

# Architecture
from models import TorchClassificationModel

# Encoders
from models.encoders import ResNet50DinoEncoder, ResNet50Encoder

# Bottlenecks
from models.bottlenecks import VectorQuantizer, DKVB

# Optmizers
from optimizers import TorchAdamWOptimizer

# Datasets
from train_data_sources import GtzanDataSource

# Data Transforms
from train_data_transforms import SimpleMusicPipeline

# Trainers
from trainers import ContinualLearningTrainer, DkvbContinualLearningTrainer

# Evaluator
from evaluators import ContinualLearningTasksEvaluatorV2

from copy import deepcopy


###############################################################
###########                SCENARIOS                ###########
###############################################################

all_tasks = scenario1 = [
    ["blues", "classical"],
    ["country", "disco"],
    ["hiphop", "jazz"],
    ["metal", "pop"],
    ["reggae", "rock"],
]

###############################################################
###########           GENERIC COMPONENTS            ###########
###############################################################

num_cross_val_splits = 5
num_epochs = 200
early_stopping_patience = 40
early_stopping_metric = "F1 Score"
epochs_keys_init = 10
num_classes = 10
data_transform = SimpleMusicPipeline(
    sample_rate=20050,
    n_fft=1024,
    win_length=None,
    hop_length=512,
    n_mels=128,
)
vector_quantizer = VectorQuantizer(
    embedding_dim=2048,
    codes_per_codebook=4096,
    num_codebooks=256,
    vq_decay=0.95,
    threshold_ema_dead_code=1e-4,
)
dkvb = DKVB(
    embedding_dim=2048,
    codes_per_codebook=4096,
    num_codebooks=256,
    vq_decay=0.95,
    threshold_ema_dead_code=1e-4,
)
###############################################################
###########                BASELINES                ###########
###############################################################

gtzan_resnet50_joint = {
    "experiment_name": "gtzan_resnet50_joint",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": ContinualLearningTrainer(
            tasks=["all"],
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
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
                    encoder=ResNet50Encoder(pretrained=True),
                    num_classes=num_classes,
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=["all"],
            test_tasks=all_tasks,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

gtzan_resnet50dino_joint = {
    "experiment_name": "gtzan_resnet50dino_joint",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": ContinualLearningTrainer(
            tasks=["all"],
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
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
                    encoder=ResNet50DinoEncoder(pretrained=True),
                    num_classes=num_classes,
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=["all"],
            test_tasks=all_tasks,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True), num_classes=num_classes
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

# Frozen Encoder

gtzan_frozenresnet50_joint = {
    "experiment_name": "gtzan_frozenresnet50_joint",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": ContinualLearningTrainer(
            tasks=["all"],
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
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
                    encoder=ResNet50Encoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=["all"],
            test_tasks=all_tasks,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

gtzan_frozenresnet50dino_joint = {
    "experiment_name": "gtzan_frozenresnet50dino_joint",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": ContinualLearningTrainer(
            tasks=["all"],
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
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
                    encoder=ResNet50DinoEncoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=["all"],
            test_tasks=all_tasks,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True), num_classes=num_classes
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

# Vector Quantizer

gtzan_vqresnet50_joint = {
    "experiment_name": "gtzan_vqresnet50_joint",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": DkvbContinualLearningTrainer(
            tasks=["all"],
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            epochs_keys_init=epochs_keys_init,
            looper=DkvbMusicGenderClassificationLooper(
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
                    encoder=ResNet50Encoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                    bottleneck=deepcopy(vector_quantizer),
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=["all"],
            test_tasks=all_tasks,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
                frozen_encoder=True,
                bottleneck=deepcopy(vector_quantizer),
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

gtzan_vqresnet50dino_joint = {
    "experiment_name": "gtzan_vqresnet50dino_joint",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": DkvbContinualLearningTrainer(
            tasks=["all"],
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            epochs_keys_init=epochs_keys_init,
            looper=DkvbMusicGenderClassificationLooper(
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
                    encoder=ResNet50DinoEncoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                    bottleneck=deepcopy(vector_quantizer),
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=["all"],
            test_tasks=all_tasks,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
                frozen_encoder=True,
                bottleneck=deepcopy(vector_quantizer),
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

# Discrete Key Value Bottleneck

gtzan_dkvbresnet50_joint = {
    "experiment_name": "gtzan_dkvbresnet50_joint",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": DkvbContinualLearningTrainer(
            tasks=["all"],
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            epochs_keys_init=epochs_keys_init,
            freeze_decoder_after_first_epoch=True,
            looper=DkvbMusicGenderClassificationLooper(
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
                    encoder=ResNet50Encoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                    bottleneck=deepcopy(dkvb),
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=["all"],
            test_tasks=all_tasks,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
                frozen_encoder=True,
                bottleneck=deepcopy(dkvb),
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

gtzan_dkvbresnet50dino_joint = {
    "experiment_name": "gtzan_dkvbresnet50dino_joint",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": DkvbContinualLearningTrainer(
            tasks=["all"],
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            epochs_keys_init=epochs_keys_init,
            freeze_decoder_after_first_epoch=True,
            looper=DkvbMusicGenderClassificationLooper(
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
                    encoder=ResNet50DinoEncoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                    bottleneck=deepcopy(dkvb),
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=["all"],
            test_tasks=all_tasks,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
                frozen_encoder=True,
                bottleneck=deepcopy(dkvb),
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}


###############################################################
###########                CONTINUAL                ###########
###############################################################

gtzan_resnet50_scenario1 = {
    "experiment_name": "gtzan_resnet50_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Naive",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": ContinualLearningTrainer(
            tasks=scenario1,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
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
                    encoder=ResNet50Encoder(pretrained=True),
                    num_classes=num_classes,
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=scenario1,
            test_tasks=scenario1,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True), num_classes=num_classes
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

gtzan_resnet50dino_scenario1 = {
    "experiment_name": "gtzan_resnet50dino_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Naive",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": ContinualLearningTrainer(
            tasks=scenario1,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
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
                    encoder=ResNet50DinoEncoder(pretrained=True),
                    num_classes=num_classes,
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=scenario1,
            test_tasks=scenario1,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True), num_classes=num_classes
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

# Frozen Encoder

gtzan_frozenresnet50_scenario1 = {
    "experiment_name": "gtzan_resnet50_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Naive",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": ContinualLearningTrainer(
            tasks=scenario1,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
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
                    encoder=ResNet50Encoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=scenario1,
            test_tasks=scenario1,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True), num_classes=num_classes
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

gtzan_frozenresnet50dino_scenario1 = {
    "experiment_name": "gtzan_frozenresnet50dino_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Naive",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": ContinualLearningTrainer(
            tasks=scenario1,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
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
                    encoder=ResNet50DinoEncoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=scenario1,
            test_tasks=scenario1,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True), num_classes=num_classes
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}


# Vector Quantizer

gtzan_vqresnet50_scenario1 = {
    "experiment_name": "gtzan_vqresnet50_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": DkvbContinualLearningTrainer(
            tasks=scenario1,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            epochs_keys_init=epochs_keys_init,
            looper=DkvbMusicGenderClassificationLooper(
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
                    encoder=ResNet50Encoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                    bottleneck=deepcopy(vector_quantizer),
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=scenario1,
            test_tasks=scenario1,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
                frozen_encoder=True,
                bottleneck=deepcopy(vector_quantizer),
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

gtzan_vqresnet50dino_scenario1 = {
    "experiment_name": "gtzan_vqresnet50dino_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": DkvbContinualLearningTrainer(
            tasks=scenario1,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            epochs_keys_init=epochs_keys_init,
            looper=DkvbMusicGenderClassificationLooper(
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
                    encoder=ResNet50DinoEncoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                    bottleneck=deepcopy(vector_quantizer),
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=scenario1,
            test_tasks=scenario1,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
                frozen_encoder=True,
                bottleneck=deepcopy(vector_quantizer),
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

# Discrete Key Value Bottleneck

gtzan_dkvbresnet50_scenario1 = {
    "experiment_name": "gtzan_dkvbresnet50_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": DkvbContinualLearningTrainer(
            tasks=scenario1,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            epochs_keys_init=epochs_keys_init,
            freeze_decoder_after_first_epoch=True,
            looper=DkvbMusicGenderClassificationLooper(
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
                    encoder=ResNet50Encoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                    bottleneck=deepcopy(dkvb),
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=scenario1,
            test_tasks=scenario1,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
                frozen_encoder=True,
                bottleneck=deepcopy(dkvb),
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}

gtzan_dkvbresnet50dino_scenario1 = {
    "experiment_name": "gtzan_dkvbresnet50dino_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": DkvbContinualLearningTrainer(
            tasks=scenario1,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            epochs_keys_init=epochs_keys_init,
            freeze_decoder_after_first_epoch=True,
            looper=DkvbMusicGenderClassificationLooper(
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
                    encoder=ResNet50DinoEncoder(pretrained=True),
                    num_classes=num_classes,
                    frozen_encoder=True,
                    bottleneck=deepcopy(dkvb),
                ),
                criteria=TorchCrossEntropyCriteria(),
                optimizer=TorchAdamWOptimizer(lr=3e-4),
                metrics={
                    "train": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                    "val": {
                        "F1 Score": F1Score(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Precision": Precision(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                        "Recall": Recall(
                            task="multiclass", average="micro", num_classes=num_classes
                        ),
                    },
                },
                experiment_tracker=TensorboardExperimentTracker(),
                model_saver=MusicGenderClassificationModelSaver(),
            ),
        ),
    },
    "evaluate": {
        "evaluator": ContinualLearningTasksEvaluatorV2(
            train_tasks=scenario1,
            test_tasks=scenario1,
            model=TorchClassificationModel(
                encoder=ResNet50Encoder(pretrained=True),
                num_classes=num_classes,
                bottleneck=deepcopy(dkvb),
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
                "F1 Score": F1Score(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Precision": Precision(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
                "Recall": Recall(
                    task="multiclass", average="micro", num_classes=num_classes
                ),
            },
            experiment_tracker=DataframeExperimentTracker(),
        ),
    },
}
