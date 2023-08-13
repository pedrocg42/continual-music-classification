from copy import deepcopy

from experiments.components import *

###############################################################
###########                SCENARIOS                ###########
###############################################################

all_tasks = [
    ["female1", "female2", "male1", "male2"],
    ["female3", "female4", "male3", "male4"],
    ["female5", "female6", "male5", "male6"],
    ["female7", "female8", "male7", "male8"],
    ["female9", "male9", "male10", "male11"],
]

scenario1 = [
    ["female1", "female2", "male1", "male2"],
    ["female3", "female4", "male3", "male4"],
    ["female5", "female6", "male5", "male6"],
    ["female7", "female8", "male7", "male8"],
    ["female9", "male9", "male10", "male11"],
]

scenario2 = [
    ["female9", "male3", "female3", "female1"],
    ["female8", "male1", "male9", "female5"],
    ["female2", "male8", "female6", "male6"],
    ["male7", "male4", "male2", "female7"],
    ["male10", "female4", "male5", "male11"],
]

scenario3 = [
    [
        ["female8", "male7", "male8", "female1"],
        ["male10", "female7", "male6", "male1"],
        ["female9", "female5", "male9", "female4"],
        ["female3", "male4", "male5", "female6"],
        ["male11", "female2", "male2", "male3"],
    ]
]

###############################################################
###########               COMPONENTS                ###########
###############################################################

num_classes = 20

# Data sources
train_vocalsetsinger_data_source = {
    "name": "VocalSetSingerDataSource",
    "args": {
        "split": "train",
        "chunk_length": 3,
        "is_eval": False,
    },
}
val_vocalsetsinger_data_source = deepcopy(train_vocalsetsinger_data_source)
val_vocalsetsinger_data_source["args"]["split"] = "val"
val_vocalsetsinger_data_source["args"]["is_eval"] = True
test_vocalsetsinger_data_source = deepcopy(train_vocalsetsinger_data_source)
test_vocalsetsinger_data_source["args"]["split"] = "test"
test_vocalsetsinger_data_source["args"]["is_eval"] = True

# Metrics
genre_classification_metrics_vocalsetsinger = [
    {
        "name": "Accuracy",
        "args": {
            "task": "multiclass",
            "average": "micro",
            "num_classes": num_classes,
        },
    },
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


###########                TRAINERS                 ###########

oracle_train_model_vocalsetsinger = {
    "name": "TorchMertClassificationModel",
    "args": {"num_classes": num_classes},
}

## Oracle
oracle_trainer_vocalsetsinger = deepcopy(oracle_trainer)
oracle_trainer_vocalsetsinger["args"]["train_model"] = oracle_train_model_vocalsetsinger
oracle_trainer_vocalsetsinger["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
oracle_trainer_vocalsetsinger["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
oracle_trainer_vocalsetsinger["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger


## Finetuning
continual_learning_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_trainer
)
continual_learning_trainer_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_trainer_vocalsetsinger_scenario1["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
continual_learning_trainer_vocalsetsinger_scenario1["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
continual_learning_trainer_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_trainer_vocalsetsinger_scenario1
)
continual_learning_trainer_vocalsetsinger_scenario2["args"]["tasks"] = scenario2

continual_learning_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_trainer_vocalsetsinger_scenario1
)
continual_learning_trainer_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


## Replay
continual_learning_replay_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_replay_trainer
)
continual_learning_replay_trainer_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_replay_trainer_vocalsetsinger_scenario1["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
continual_learning_replay_trainer_vocalsetsinger_scenario1["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
continual_learning_replay_trainer_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_replay_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_replay_trainer_vocalsetsinger_scenario1
)
continual_learning_replay_trainer_vocalsetsinger_scenario2["args"]["tasks"] = scenario2

continual_learning_replay_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_replay_trainer_vocalsetsinger_scenario1
)
continual_learning_replay_trainer_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


## iCaRL
continual_learning_icarl_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_icarl_trainer
)
continual_learning_icarl_trainer_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_icarl_trainer_vocalsetsinger_scenario1["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
continual_learning_icarl_trainer_vocalsetsinger_scenario1["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
continual_learning_icarl_trainer_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_icarl_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_icarl_trainer_vocalsetsinger_scenario1
)
continual_learning_icarl_trainer_vocalsetsinger_scenario2["args"]["tasks"] = scenario2

continual_learning_icarl_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_icarl_trainer_vocalsetsinger_scenario1
)
continual_learning_icarl_trainer_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


# VQ
continual_learning_vq_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_vq_trainer
)
continual_learning_vq_trainer_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_trainer_vocalsetsinger_scenario1["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
continual_learning_vq_trainer_vocalsetsinger_scenario1["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
continual_learning_vq_trainer_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_vq_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_vq_trainer_vocalsetsinger_scenario1
)
continual_learning_vq_trainer_vocalsetsinger_scenario2["args"]["tasks"] = scenario2

continual_learning_vq_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_vq_trainer_vocalsetsinger_scenario1
)
continual_learning_vq_trainer_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


# DKVB
continual_learning_dkvb_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_dkvb_trainer
)
continual_learning_dkvb_trainer_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_trainer_vocalsetsinger_scenario1["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
continual_learning_dkvb_trainer_vocalsetsinger_scenario1["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
continual_learning_dkvb_trainer_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_dkvb_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_dkvb_trainer_vocalsetsinger_scenario1
)
continual_learning_dkvb_trainer_vocalsetsinger_scenario2["args"]["tasks"] = scenario2

continual_learning_dkvb_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_dkvb_trainer_vocalsetsinger_scenario1
)
continual_learning_dkvb_trainer_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


## GEM
continual_learning_gem_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_gem_trainer
)
continual_learning_gem_trainer_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_gem_trainer_vocalsetsinger_scenario1["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
continual_learning_gem_trainer_vocalsetsinger_scenario1["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
continual_learning_gem_trainer_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_gem_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_gem_trainer_vocalsetsinger_scenario1
)
continual_learning_gem_trainer_vocalsetsinger_scenario2["args"]["tasks"] = scenario2

continual_learning_gem_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_gem_trainer_vocalsetsinger_scenario1
)
continual_learning_gem_trainer_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


## EWC
continual_learning_ewc_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_ewc_trainer
)
continual_learning_ewc_trainer_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_ewc_trainer_vocalsetsinger_scenario1["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
continual_learning_ewc_trainer_vocalsetsinger_scenario1["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
continual_learning_ewc_trainer_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_ewc_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_ewc_trainer_vocalsetsinger_scenario1
)
continual_learning_ewc_trainer_vocalsetsinger_scenario2["args"]["tasks"] = scenario2

continual_learning_ewc_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_ewc_trainer_vocalsetsinger_scenario1
)
continual_learning_ewc_trainer_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


## L2P
continual_learning_l2p_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_l2p_trainer
)
continual_learning_l2p_trainer_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_trainer_vocalsetsinger_scenario1["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
continual_learning_l2p_trainer_vocalsetsinger_scenario1["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
continual_learning_l2p_trainer_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_l2p_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_l2p_trainer_vocalsetsinger_scenario1
)
continual_learning_l2p_trainer_vocalsetsinger_scenario2["args"]["tasks"] = scenario2

continual_learning_l2p_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_l2p_trainer_vocalsetsinger_scenario1
)
continual_learning_l2p_trainer_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


## L2Center
continual_learning_l2center_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_l2center_trainer
)
continual_learning_l2center_trainer_vocalsetsinger_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_l2center_trainer_vocalsetsinger_scenario1["args"][
    "train_data_source"
] = train_vocalsetsinger_data_source
continual_learning_l2center_trainer_vocalsetsinger_scenario1["args"][
    "val_data_source"
] = val_vocalsetsinger_data_source
continual_learning_l2center_trainer_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_l2center_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_l2center_trainer_vocalsetsinger_scenario1
)
continual_learning_l2center_trainer_vocalsetsinger_scenario2["args"][
    "tasks"
] = scenario2

continual_learning_l2center_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_l2center_trainer_vocalsetsinger_scenario1
)
continual_learning_l2center_trainer_vocalsetsinger_scenario3["args"][
    "tasks"
] = scenario3

# CosineCenter
continual_learning_cosinecenter_trainer_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_l2center_trainer_vocalsetsinger_scenario1
)
continual_learning_cosinecenter_trainer_vocalsetsinger_scenario1["args"][
    "train_model"
] = train_model_cosinecenter

continual_learning_cosinecenter_trainer_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_cosinecenter_trainer_vocalsetsinger_scenario1
)
continual_learning_cosinecenter_trainer_vocalsetsinger_scenario2["args"][
    "tasks"
] = scenario2

continual_learning_cosinecenter_trainer_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_cosinecenter_trainer_vocalsetsinger_scenario1
)
continual_learning_cosinecenter_trainer_vocalsetsinger_scenario3["args"][
    "tasks"
] = scenario3


###########               EVALUATORS                ###########

# Oracle
oracle_evaluator_vocalsetsinger = deepcopy(oracle_evaluator)
oracle_evaluator_vocalsetsinger["args"]["model"] = oracle_train_model_vocalsetsinger
oracle_evaluator_vocalsetsinger["args"]["tasks"] = all_tasks
oracle_evaluator_vocalsetsinger["args"]["data_source"] = test_vocalsetsinger_data_source
oracle_evaluator_vocalsetsinger["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

oracle_evaluator_vocalsetsinger_scenario1 = deepcopy(oracle_evaluator_vocalsetsinger)
oracle_evaluator_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
oracle_evaluator_vocalsetsinger_scenario2 = deepcopy(oracle_evaluator_vocalsetsinger)
oracle_evaluator_vocalsetsinger_scenario2["args"]["tasks"] = scenario2
oracle_evaluator_vocalsetsinger_scenario3 = deepcopy(oracle_evaluator_vocalsetsinger)
oracle_evaluator_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


# Finetuning
continual_learning_evaluator_vocalsetsinger_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_evaluator_vocalsetsinger_scenario1["args"][
    "data_source"
] = test_vocalsetsinger_data_source
continual_learning_evaluator_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_evaluator_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_evaluator_vocalsetsinger_scenario1
)
continual_learning_evaluator_vocalsetsinger_scenario2["args"]["tasks"] = scenario2
continual_learning_evaluator_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_evaluator_vocalsetsinger_scenario1
)
continual_learning_evaluator_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


## VQ
continual_learning_vq_evaluator_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_evaluator_vq
)
continual_learning_vq_evaluator_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_evaluator_vocalsetsinger_scenario1["args"][
    "data_source"
] = test_vocalsetsinger_data_source
continual_learning_vq_evaluator_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_vq_evaluator_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_vq_evaluator_vocalsetsinger_scenario1
)
continual_learning_vq_evaluator_vocalsetsinger_scenario2["args"]["tasks"] = scenario2
continual_learning_vq_evaluator_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_vq_evaluator_vocalsetsinger_scenario1
)
continual_learning_vq_evaluator_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


## DKVB
continual_learning_dkvb_evaluator_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_evaluator_dkvb
)
continual_learning_dkvb_evaluator_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_evaluator_vocalsetsinger_scenario1["args"][
    "data_source"
] = test_vocalsetsinger_data_source
continual_learning_dkvb_evaluator_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_dkvb_evaluator_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_dkvb_evaluator_vocalsetsinger_scenario1
)
continual_learning_dkvb_evaluator_vocalsetsinger_scenario2["args"]["tasks"] = scenario2
continual_learning_dkvb_evaluator_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_dkvb_evaluator_vocalsetsinger_scenario1
)
continual_learning_dkvb_evaluator_vocalsetsinger_scenario3["args"]["tasks"] = scenario3


# L2P
continual_learning_l2p_evaluator_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_evaluator_l2p
)
continual_learning_l2p_evaluator_vocalsetsinger_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_evaluator_vocalsetsinger_scenario1["args"][
    "data_source"
] = test_vocalsetsinger_data_source
continual_learning_l2p_evaluator_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_l2p_evaluator_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_l2p_evaluator_vocalsetsinger_scenario1
)
continual_learning_l2p_evaluator_vocalsetsinger_scenario2["args"]["tasks"] = scenario2
continual_learning_l2p_evaluator_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_l2p_evaluator_vocalsetsinger_scenario1
)
continual_learning_l2p_evaluator_vocalsetsinger_scenario3["args"]["tasks"] = scenario3

## L2Center
continual_learning_l2center_evaluator_vocalsetsinger_scenario1 = deepcopy(
    continual_learning_evaluator_l2center
)
continual_learning_l2center_evaluator_vocalsetsinger_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_l2center_evaluator_vocalsetsinger_scenario1["args"][
    "data_source"
] = test_vocalsetsinger_data_source
continual_learning_l2center_evaluator_vocalsetsinger_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsinger

continual_learning_l2center_evaluator_vocalsetsinger_scenario2 = deepcopy(
    continual_learning_l2center_evaluator_vocalsetsinger_scenario1
)
continual_learning_l2center_evaluator_vocalsetsinger_scenario2["args"][
    "tasks"
] = scenario2
continual_learning_l2center_evaluator_vocalsetsinger_scenario3 = deepcopy(
    continual_learning_l2center_evaluator_vocalsetsinger_scenario1
)
continual_learning_l2center_evaluator_vocalsetsinger_scenario3["args"][
    "tasks"
] = scenario3


###############################################################
###########               EXPERIMENTS               ###########
###############################################################

###########                BASELINES                ###########

mert95m_base_oracle_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_base_oracle_vocalsetsinger_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_vocalsetsinger,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_base_oracle_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_base_oracle_vocalsetsinger_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_vocalsetsinger,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_vocalsetsinger_scenario2,
    },
}


mert95m_base_oracle_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_base_oracle_vocalsetsinger_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_vocalsetsinger,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_vocalsetsinger_scenario3,
    },
}


###########            CONTINUAL LEARNING           ###########

# SCENARIO 1

mert95m_finetuning_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_finetuning_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_replay_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_replay_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_icarl_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_icarl_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_vq_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_vq_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_dkvb_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_dkvb_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_gem_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_gem_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_ewc_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_ewc_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_l2p_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_l2p_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_l2center_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_l2center_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsetsinger_scenario1,
    },
}

mert95m_cosinecenter_cl_vocalsetsinger_scenario1 = {
    "experiment_name": "mert95m_cosinecenter_cl_vocalsetsinger_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "CosineCenter",
    # data
    "train": {
        "trainer": continual_learning_cosinecenter_trainer_vocalsetsinger_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsetsinger_scenario1,
    },
}

# SCENARIO 2

mert95m_finetuning_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_finetuning_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario2,
    },
}

mert95m_replay_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_replay_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario2,
    },
}

mert95m_icarl_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_icarl_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario2,
    },
}

mert95m_vq_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_vq_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_vocalsetsinger_scenario2,
    },
}

mert95m_dkvb_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_dkvb_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_vocalsetsinger_scenario2,
    },
}

mert95m_gem_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_gem_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario2,
    },
}

mert95m_ewc_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_ewc_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario2,
    },
}

mert95m_l2p_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_l2p_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_vocalsetsinger_scenario2,
    },
}

mert95m_l2center_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_l2center_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsetsinger_scenario2,
    },
}

mert95m_cosinecenter_cl_vocalsetsinger_scenario2 = {
    "experiment_name": "mert95m_cosinecenter_cl_vocalsetsinger_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "CosineCenter",
    # data
    "train": {
        "trainer": continual_learning_cosinecenter_trainer_vocalsetsinger_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsetsinger_scenario2,
    },
}


# SCENARIO 3

mert95m_finetuning_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_finetuning_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario3,
    },
}

mert95m_replay_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_replay_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario3,
    },
}

mert95m_icarl_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_icarl_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario3,
    },
}

mert95m_vq_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_vq_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_vocalsetsinger_scenario3,
    },
}

mert95m_dkvb_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_dkvb_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_vocalsetsinger_scenario3,
    },
}

mert95m_gem_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_gem_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario3,
    },
}

mert95m_ewc_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_ewc_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsinger_scenario3,
    },
}

mert95m_l2p_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_l2p_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_vocalsetsinger_scenario3,
    },
}

mert95m_l2center_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_l2center_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsetsinger_scenario3,
    },
}

mert95m_cosinecenter_cl_vocalsetsinger_scenario3 = {
    "experiment_name": "mert95m_cosinecenter_cl_vocalsetsinger_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "CosineCenter",
    # data
    "train": {
        "trainer": continual_learning_cosinecenter_trainer_vocalsetsinger_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsetsinger_scenario3,
    },
}
