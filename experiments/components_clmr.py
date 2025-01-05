from copy import deepcopy

###############################################################
###########           GENERIC COMPONENTS            ###########
###############################################################

# Training parameters
num_epochs = 200
early_stopping_patience = 5
early_stopping_metric = "Accuracy"
epochs_keys_init = 10
batch_size = 64


# Data transforms
mert_data_transform = {
    "name": "ResamplerDataTransform",
    "args": {
        "input_sample_rate": 22050,
        "output_sample_rate": 22050,
    },
}

mert_data_transform_vocalset = {
    "name": "ResamplerDataTransform",
    "args": {
        "input_sample_rate": 44100,
        "output_sample_rate": 22050,
    },
}

mert_data_transform_nsynth = {
    "name": "ResamplerDataTransform",
    "args": {
        "input_sample_rate": 16000,
        "output_sample_rate": 22050,
    },
}


# Train models
train_model = {
    "name": "TorchClmrClassIncrementalModel",
    "args": {
        "encoder": {
            "name": "ClmrEncoder",
            "args": {
                "pretrained": True,
            },
        },
        "frozen_encoder": True,
    },
}

train_model_l2center = {
    "name": "TorchEmbeddingModel",
    "args": {
        "encoder": {
            "name": "ClmrEncoder",
            "args": {
                "pretrained": True,
            },
        },
        "frozen_encoder": True,
        "average_hidden": False,
    },
}

# Trainers
trainer = {
    "name": "ClassIncrementalLearningTrainer",
    "args": {
        "tasks": None,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": early_stopping_metric,
        "train_data_source": None,
        "val_data_source": None,
        "train_data_transform": mert_data_transform,
        "val_data_transform": mert_data_transform,
        "train_model": None,
        "metrics_config": None,
        "experiment_tracker": {"name": "TensorboardExperimentTracker"},
        "model_saver": {"name": "MusicGenreClassificationModelSaver"},
        "looper": {
            "name": "MusicGenreClassificationLooper",
            "args": {
                "criteria": {"name": "TorchCrossEntropyCriteria"},
                "optimizer": {"name": "TorchAdamOptimizer"},
            },
        },
    },
}

oracle_trainer = deepcopy(trainer)
oracle_trainer["name"] = "ContinualLearningTrainer"
oracle_trainer["args"]["tasks"] = ["all"]

continual_learning_trainer = deepcopy(trainer)
continual_learning_trainer["args"]["train_model"] = train_model

## Replay
continual_learning_replay_trainer = deepcopy(trainer)
continual_learning_replay_trainer["name"] = "ReplayContinualLearningTrainer"
continual_learning_replay_trainer["args"]["train_model"] = train_model
continual_learning_replay_trainer["args"]["num_memories"] = 100

## Replay
continual_learning_icarl_trainer = deepcopy(trainer)
continual_learning_icarl_trainer["name"] = "iCaRLContinualLearningTrainer"
continual_learning_icarl_trainer["args"]["train_model"] = train_model
continual_learning_icarl_trainer["args"]["num_memories"] = 100
continual_learning_icarl_trainer["args"]["looper"] = {
    "name": "iCaRLMusicGenreClassificationLooper",
    "args": {
        "criteria": {"name": "TorchCrossEntropyCriteria"},
        "optimizer": {"name": "TorchAdamOptimizer"},
        "T": 2.0,
    },
}

## GEM
continual_learning_gem_trainer = deepcopy(trainer)
continual_learning_gem_trainer["name"] = "GemContinualLearningTrainer"
continual_learning_gem_trainer["args"]["looper"]["name"] = "GemMusicGenreClassificationLooper"
continual_learning_gem_trainer["args"]["looper"]["args"]["optimizer"] = {
    "name": "GemOptimizer",
    "args": {"num_memories": 100, "memory_strength": 0.5},
}
continual_learning_gem_trainer["args"]["train_model"] = train_model

## EWC
continual_learning_ewc_trainer = deepcopy(trainer)
continual_learning_ewc_trainer["name"] = "EwcContinualLearningTrainer"
continual_learning_ewc_trainer["args"]["looper"]["name"] = "EwcMusicGenreClassificationLooper"
continual_learning_ewc_trainer["args"]["looper"]["args"]["optimizer"] = {
    "name": "EwcOptimizer",
    "args": {"ewc_lambda": 0.1},
}
continual_learning_ewc_trainer["args"]["train_model"] = train_model

## Embedding center
continual_learning_l2center_trainer = deepcopy(trainer)
continual_learning_l2center_trainer["name"] = "ContinualLearningTrainerL2Center"
continual_learning_l2center_trainer["args"]["train_model"] = train_model_l2center
continual_learning_l2center_trainer["args"]["looper"] = {
    "name": "MusicContinualLearningEmbeddingLooper",
}

# Evaluators
evaluator = {
    "name": "ClassIncrementalLearningEvaluator",
    "args": {
        "tasks": None,
        "model": train_model,
        "model_saver": {"name": "MusicGenreClassificationModelSaver"},
        "data_source": None,
        "data_transform": mert_data_transform,
        "metrics_config": None,
        "experiment_tracker": {"name": "DataframeExperimentTracker"},
    },
}

oracle_evaluator = deepcopy(evaluator)
oracle_evaluator["name"] = "ClassIncrementalLearningOracleEvaluator"
oracle_evaluator["args"]["tasks"] = ["all"]

continual_learning_evaluator_l2center = deepcopy(evaluator)
continual_learning_evaluator_l2center["args"]["model"] = train_model_l2center
continual_learning_evaluator_l2center["name"] = "ClassIncrementalLearningL2CenterEvaluator"
