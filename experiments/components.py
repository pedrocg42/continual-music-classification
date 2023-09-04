from copy import deepcopy

###############################################################
###########           GENERIC COMPONENTS            ###########
###############################################################

# Training parameters
num_epochs = 200
early_stopping_patience = 20
early_stopping_metric = "Accuracy"
epochs_keys_init = 10
batch_size = 64


# Data transforms
mert_data_transform = {
    "name": "MertDataTransform",
    "args": {
        "input_sample_rate": 22050,
        "output_sample_rate": 24000,
    },
}

mert_data_transform_vocalset = {
    "name": "MertDataTransform",
    "args": {
        "input_sample_rate": 44100,
        "output_sample_rate": 24000,
    },
}

mert_data_transform_nsynth = {
    "name": "MertDataTransform",
    "args": {
        "input_sample_rate": 16000,
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
    },
}

train_model_l2p = {
    "name": "TorchL2PClassIncrementalModel",
    "args": {
        "encoder": {
            "name": "MertEncoderL2P",
            "args": {
                "pretrained": True,
                "prompt_pool_size": 20,  # M
                "prompt_length": 5,  # L_p
                "selection_size": 8,  # N
            },
        },
        "frozen_encoder": False,
    },
}

train_model_l2center = {
    "name": "TorchEmbeddingModel",
    "args": {
        "encoder": {
            "name": "MertEncoder",
            "args": {
                "pretrained": True,
            },
        },
        "frozen_encoder": True,
    },
}

train_model_cosinecenter = {
    "name": "TorchEmbeddingCosineModel",
    "args": {
        "encoder": {
            "name": "MertEncoder",
            "args": {
                "pretrained": True,
            },
        },
        "frozen_encoder": True,
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
    "args": {"num_memories": 100, "memory_strength": 0.5},
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

## L2P
continual_learning_l2p_trainer = deepcopy(trainer)
continual_learning_l2p_trainer["args"]["train_model"] = train_model_l2p
continual_learning_l2p_trainer["args"]["batch_size"] = 16
continual_learning_l2p_trainer["args"]["looper"] = {
    "name": "L2PMusicGenreClassificationLooper",
    "args": {
        "criteria": {"name": "TorchCrossEntropyCriteria"},
        "optimizer": {"name": "TorchAdamOptimizer"},
        "lamb": 0.5,
    },
}

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

continual_learning_evaluator_vq = deepcopy(evaluator)
continual_learning_evaluator_vq["args"]["model"] = train_model_vq
continual_learning_evaluator_vq["name"] = "ClassIncrementalLearningDKVBEvaluator"

continual_learning_evaluator_dkvb = deepcopy(evaluator)
continual_learning_evaluator_dkvb["args"]["model"] = train_model_dkvb
continual_learning_evaluator_dkvb["name"] = "ClassIncrementalLearningDKVBEvaluator"

continual_learning_evaluator_l2p = deepcopy(evaluator)
continual_learning_evaluator_l2p["args"]["model"] = train_model_l2p
continual_learning_evaluator_l2p["name"] = "ClassIncrementalLearningL2PEvaluator"

continual_learning_evaluator_l2center = deepcopy(evaluator)
continual_learning_evaluator_l2center["args"]["model"] = train_model_l2center
continual_learning_evaluator_l2center[
    "name"
] = "ClassIncrementalLearningL2CenterEvaluator"

continual_learning_evaluator_cosinecenter = deepcopy(evaluator)
continual_learning_evaluator_cosinecenter["args"]["model"] = train_model_cosinecenter
continual_learning_evaluator_cosinecenter[
    "name"
] = "ClassIncrementalLearningL2CenterEvaluator"
