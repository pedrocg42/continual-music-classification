from copy import deepcopy

###############################################################
###########           GENERIC COMPONENTS            ###########
###############################################################

# Training parameters
num_cross_val_splits = 5
num_epochs = 2
early_stopping_patience = 40
early_stopping_metric = "F1 Score"
epochs_keys_init = 10
num_classes = 10
batch_size = 8

# Data sources
train_gtzan_data_source = {
    "name": "GtzanDataSource",
    "args": {
        "split": "train",
        "num_cross_val_splits": 5,
        "hop_length": 512,
        "length_spectrogram": 128,
    },
}
val_gtzan_data_source = deepcopy(train_gtzan_data_source)
val_gtzan_data_source["args"]["split"] = "val"
val_gtzan_data_source["args"]["is_eval"] = True
test_gtzan_data_source = deepcopy(train_gtzan_data_source)
test_gtzan_data_source["args"]["split"] = "test"
val_gtzan_data_source["args"]["is_eval"] = True


# Data transforms
mert_data_transform = {
    "name": "MertDataTransform",
    "args": {
        "input_sample_rate": 22050,
        "output_sample_rate": 24000,
    },
}

# Bottlenecks
vector_quantizer = {
    "VectorQuantizer": {
        "embedding_dim": 2048,
        "codes_per_codebook": 4096,
        "num_codebooks": 256,
        "vq_decay": 0.95,
        "threshold_ema_dead_code": 1e-4,
    }
}

dkvb = {
    "name": "DKVB",
    "args": {
        "embedding_dim": 2048,
        "codes_per_codebook": 4096,
        "num_codebooks": 256,
        "vq_decay": 0.95,
        "threshold_ema_dead_code": 1e-4,
    },
}


# Train models
train_model = {
    "name": "TorchClassificationModel",
    "args": {
        "encoder": {
            "name": "MertEncoder",
            "args": {
                "pretrained": True,
            },
        },
        "num_classes": num_classes,
        "freeze_encoder": True,
    },
}

train_model_vq = deepcopy(train_model)
train_model_vq["args"]["frozen_encoder"] = True
train_model_vq["args"]["bottleneck"] = vector_quantizer

train_model_dkvb = deepcopy(train_model)
train_model_dkvb["args"]["frozen_encoder"] = True
train_model_dkvb["args"]["bottleneck"] = dkvb

# Metrics
genre_classification_metrics = [
    {
        "name": "F1 Score",
        "args": {
            "task": "multiclass",
            "average": "micro",
            "num_classes": num_classes,
        },
    },
    {
        "name": "Precision",
        "args": {
            "task": "multiclass",
            "average": "micro",
            "num_classes": num_classes,
        },
    },
    {
        "name": "Recall",
        "args": {
            "task": "multiclass",
            "average": "micro",
            "num_classes": num_classes,
        },
    },
]


# Trainer
continual_learning_trainer = {
    "name": "ContinualLearningTrainer",
    "args": {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": early_stopping_metric,
        "looper": {
            "name": "MusicGenreClassificationLooper",
            "args": {
                "train_data_source": train_gtzan_data_source,
                "val_data_source": val_gtzan_data_source,
                "train_data_transform": mert_data_transform,
                "val_data_transform": mert_data_transform,
                "train_model": train_model,
                "criteria": {"name": "TorchCrossEntropyCriteria"},
                "optimizer": {"name": "TorchAdamWOptimizer"},
                "metrics": genre_classification_metrics,
                "experiment_tracker": {"name": "TensorboardExperimentTracker"},
                "model_saver": {"name": "MusicGenreClassificationModelSaver"},
            },
        },
    },
}


# Evaluator
continual_learning_evaluator = {
    "name": "ContinualLearningTasksEvaluatorV2",
    "args": {
        "train_tasks": None,
        "test_tasks": None,
        "model": train_model,
        "model_saver": {"name": "MusicGenreClassificationModelSaver"},
        "data_source": test_gtzan_data_source,
        "data_transform": mert_data_transform,
        "metrics": genre_classification_metrics,
        "experiment_tracker": {"name": "DataframeExperimentTracker"},
    },
}
