###############################################################
###########           GENERIC COMPONENTS            ###########
###############################################################

# Training parameters
num_cross_val_splits = 5
num_epochs = 200
early_stopping_patience = 40
early_stopping_metric = "F1 Score"
epochs_keys_init = 10
num_classes = 10

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
val_gtzan_data_source = train_gtzan_data_source.copy()
val_gtzan_data_source["args"]["split"] = "val"
test_gtzan_data_source = train_gtzan_data_source.copy()
test_gtzan_data_source["args"]["split"] = "test"


# Data transforms
music2vec_data_transform = {
    "name": "Music2VecDataTransform",
    "args": {
        "resample_rate": 16000,
        "sampling_rate": 16000,
        "length_spectrogram": 128,
        "hop_length": 512,
        "num_mel_bins": 64,
        "f_min": 50,
        "f_max": 8000,
        "normalize": True,
        "augment": True,
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
            "name": "Music2VecEncoder",
            "args": {
                "pretrained": True,
            },
        },
        "num_classes": num_classes,
    },
}

train_model_vq = train_model.copy()
train_model_vq["args"]["frozen_encoder"] = True
train_model_vq["args"]["bottleneck"] = vector_quantizer

train_model_dkvb = train_model.copy()
train_model_dkvb["args"]["frozen_encoder"] = True
train_model_dkvb["args"]["bottleneck"] = dkvb

# Metrics
gender_classification_metrics = (
    [
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
    ],
)

# Trainer
continual_learning_trainer = {
    "name": "ContinualLearningTrainer",
    "args": {
        "num_epochs": num_epochs,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": early_stopping_metric,
        "epochs_keys_init": epochs_keys_init,
        "looper": {
            "name": "MusicGenderClassificationLooper",
            "args": {
                "train_data_source": train_gtzan_data_source,
                "val_data_source": val_gtzan_data_source,
                "train_data_transform": music2vec_data_transform,
                "val_data_transform": music2vec_data_transform,
                "train_model": train_model,
                "criteria": {"name": "TorchCrossEntropyCriteria"},
                "optimizer": {"name": "TorchAdamWOptimizer"},
                "metrics": gender_classification_metrics,
                "experiment_tracker": {"name": "TensorboardExperimentTracker"},
                "model_saver": {"name": "MusicGenderClassificationModelSaver"},
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
        "model_saver": {"name": "MusicGenderClassificationModelSaver"},
        "data_source": test_gtzan_data_source,
        "data_transform": music2vec_data_transform,
        "metrics": gender_classification_metrics,
        "experiment_tracker": {"name": "DataframeExperimentTracker"},
    },
}
