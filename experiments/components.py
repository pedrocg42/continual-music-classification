from copy import deepcopy

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
batch_size = 32

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
test_gtzan_data_source["args"]["is_eval"] = True


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
    "name": "VectorQuantizer",
    "args": {
        "embedding_dim": 768,
        "codes_per_codebook": 64,
        "num_codebooks": 128,
        "vq_decay": 0.95,
        "threshold_ema_dead_code": 1e-4,
    },
}

dkvb = {
    "name": "DKVB",
    "args": {
        "embedding_dim": 768,
        "codes_per_codebook": 64,
        "num_codebooks": 128,
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


# Trainers
continual_learning_trainer = {
    "name": "DkvbContinualLearningTrainer",
    "args": {
        "tasks": None,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": early_stopping_metric,
        "looper": {
            "name": "DkvbMusicGenreClassificationLooper",
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

continual_learning_dkvb_trainer = deepcopy(continual_learning_trainer)
continual_learning_dkvb_trainer["args"].update(
    {
        "epochs_keys_init": 10,
        "freeze_decoder_after_first_episode": True,
    }
)
continual_learning_dkvb_trainer["args"]["looper"]["args"][
    "train_model"
] = train_model_dkvb

continual_learning_vq_trainer = deepcopy(continual_learning_trainer)
continual_learning_vq_trainer["args"].update(
    {
        "epochs_keys_init": 10,
        "freeze_decoder_after_first_episode": False,
    }
)
continual_learning_vq_trainer["args"]["looper"]["args"]["train_model"] = train_model_vq

continual_learning_gem_trainer = deepcopy(continual_learning_trainer)
continual_learning_gem_trainer["name"] = "GemContinualLearningTrainer"
continual_learning_gem_trainer["args"]["looper"][
    "name"
] = "GemMusicGenreClassificationLooper"
continual_learning_gem_trainer["args"]["looper"]["args"]["optimizer"] = {
    "name": "GemOptimizer",
    "args": {"patterns_per_experience": 32, "memory_strength": 0.5},
}


# Evaluators
continual_learning_evaluator = {
    "name": "ContinualLearningEvaluator",
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

continual_learning_evaluator_vq = deepcopy(continual_learning_evaluator)
continual_learning_evaluator_vq["args"]["model"] = train_model_vq

continual_learning_evaluator_dkvb = deepcopy(continual_learning_evaluator)
continual_learning_evaluator_dkvb["args"]["model"] = train_model_dkvb
