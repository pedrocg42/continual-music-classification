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
batch_size = 16

# Data sources
train_gtzan_data_source = {
    "name": "GtzanDataSource",
    "args": {
        "split": "train",
        "chunk_length": 5,
        "num_cross_val_splits": 5,
        "is_eval": False,
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
        "projection_embedding_dim": 8,
        "codes_per_codebook": 1024,
        "num_codebooks": 128,
        "vq_decay": 0.95,
        "threshold_ema_dead_code": 1e-4,
    },
}


# Train models
oracle_train_model = {
    "name": "TorchMertClassificationModel",
    "args": {"num_classes": num_classes},
}

train_model = {
    "name": "TorchMertClassIncrementalModel",
    "args": {},
}


train_model_vq = {
    "name": "TorchBottleneckClassificationModel",
    "args": {
        "bottleneck_config": vector_quantizer,
        "encoder": {
            "name": "MertEncoder",
            "args": {
                "pretrained": True,
            },
        },
        "frozen_encoder": True,
        "num_classes": num_classes,
    },
}

train_model_dkvb = {
    "name": "TorchBottleneckClassIncrementalModel",
    "args": {
        "bottleneck_config": dkvb,
        "encoder": {
            "name": "MertEncoder",
            "args": {
                "pretrained": True,
            },
        },
        "frozen_encoder": True,
        "num_classes": num_classes,
    },
}

train_model_l2p = {
    "name": "TorchMertClassIncrementalModel",
    "args": {
        "encoder": {
            "name": "MertEncoderL2P",
            "args": {
                "pretrained": True,
                "prompt_pool_size": 10,  # M
                "prompt_length": 5,  # L_p
                "selection_size": 5,  # N
            },
        },
        "frozen_encoder": False,
    },
}

# Metrics
genre_classification_metrics = [
    {
        "name": "F1 Score",
        "args": {
            "task": "multiclass",
            "average": "macro",
            "num_classes": num_classes,
        },
    },
    {
        "name": "Precision",
        "args": {
            "task": "multiclass",
            "average": "macro",
            "num_classes": num_classes,
        },
    },
    {
        "name": "Recall",
        "args": {
            "task": "multiclass",
            "average": "macro",
            "num_classes": num_classes,
        },
    },
]


# Trainers
trainer = {
    "name": "ClassIncrementalLearningTrainer",
    "args": {
        "tasks": None,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": early_stopping_metric,
        "train_data_source": train_gtzan_data_source,
        "val_data_source": val_gtzan_data_source,
        "train_data_transform": mert_data_transform,
        "val_data_transform": mert_data_transform,
        "train_model": None,
        "metrics_config": genre_classification_metrics,
        "experiment_tracker": {"name": "TensorboardExperimentTracker"},
        "model_saver": {"name": "MusicGenreClassificationModelSaver"},
        "looper": {
            "name": "MusicGenreClassificationLooper",
            "args": {
                "criteria": {"name": "TorchCrossEntropyCriteria"},
                "optimizer": {"name": "TorchSgdOptimizer"},
            },
        },
    },
}

oracle_trainer = deepcopy(trainer)
oracle_trainer["name"] = "ContinualLearningTrainer"
oracle_trainer["args"]["train_model"] = oracle_train_model
oracle_trainer["args"]["tasks"] = ["all"]

continual_learning_trainer = deepcopy(trainer)
continual_learning_trainer["args"]["train_model"] = train_model

## Replay
continual_learning_replay_trainer = deepcopy(trainer)
continual_learning_replay_trainer["name"] = "ReplayContinualLearningTrainer"
continual_learning_replay_trainer["args"]["train_model"] = train_model
continual_learning_replay_trainer["args"]["num_memories"] = 100

## VQ
continual_learning_vq_trainer = deepcopy(trainer)
continual_learning_vq_trainer["name"] = "DkvbContinualLearningTrainer"
continual_learning_vq_trainer["args"].update(
    {
        "epochs_keys_init": 10,
        "freeze_decoder_after_first_episode": False,
    }
)
continual_learning_vq_trainer["args"]["looper"][
    "name"
] = "DkvbMusicGenreClassificationLooper"
continual_learning_vq_trainer["args"]["train_model"] = train_model_vq

## DKVB
continual_learning_dkvb_trainer = deepcopy(trainer)
continual_learning_dkvb_trainer["name"] = "DkvbContinualLearningTrainer"
continual_learning_dkvb_trainer["args"].update(
    {
        "epochs_keys_init": 10,
        "freeze_decoder_after_first_episode": False,
    }
)
continual_learning_dkvb_trainer["args"]["looper"][
    "name"
] = "DkvbMusicGenreClassificationLooper"
continual_learning_dkvb_trainer["args"]["train_model"] = train_model_dkvb
continual_learning_dkvb_trainer["args"]["looper"]["args"]["optimizer"]["args"] = {
    "lr": 10.0
}

## GEM
continual_learning_gem_trainer = deepcopy(trainer)
continual_learning_gem_trainer["name"] = "GemContinualLearningTrainer"
continual_learning_gem_trainer["args"]["looper"][
    "name"
] = "GemMusicGenreClassificationLooper"
continual_learning_gem_trainer["args"]["looper"]["args"]["optimizer"] = {
    "name": "GemOptimizer",
    "args": {"patterns_per_experience": 10, "memory_strength": 0.5},
}
continual_learning_gem_trainer["args"]["train_model"] = train_model

## EWC
continual_learning_ewc_trainer = deepcopy(trainer)
continual_learning_ewc_trainer["name"] = "EwcContinualLearningTrainer"
continual_learning_ewc_trainer["args"]["looper"][
    "name"
] = "EwcMusicGenreClassificationLooper"
continual_learning_ewc_trainer["args"]["looper"]["args"]["optimizer"] = {
    "name": "EwcOptimizer",
    "args": {"ewc_lambda": 0.1},
}
continual_learning_ewc_trainer["args"]["train_model"] = train_model

## VQ
continual_learning_l2p_trainer = deepcopy(trainer)
continual_learning_l2p_trainer["args"]["train_model"] = train_model_l2p

# Evaluators
evaluator = {
    "name": "ClassIncrementalLearningEvaluator",
    "args": {
        "tasks": None,
        "model": train_model,
        "model_saver": {"name": "MusicGenreClassificationModelSaver"},
        "data_source": test_gtzan_data_source,
        "data_transform": mert_data_transform,
        "metrics_config": genre_classification_metrics,
        "experiment_tracker": {"name": "DataframeExperimentTracker"},
    },
}

oracle_evaluator = deepcopy(evaluator)
oracle_evaluator["name"] = "ClassIncrementalLearningOracleEvaluator"
oracle_evaluator["args"]["model"] = oracle_train_model
oracle_evaluator["args"]["tasks"] = ["all"]

continual_learning_evaluator_vq = deepcopy(evaluator)
continual_learning_evaluator_vq["args"]["model"] = train_model_vq
continual_learning_evaluator_vq["name"] = "ClassIncrementalLearningDKVBEvaluator"

continual_learning_evaluator_dkvb = deepcopy(evaluator)
continual_learning_evaluator_dkvb["args"]["model"] = train_model_dkvb
continual_learning_evaluator_dkvb["name"] = "ClassIncrementalLearningDKVBEvaluator"
