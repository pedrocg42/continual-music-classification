from copy import deepcopy

from experiments.components import *

###############################################################
###########                SCENARIOS                ###########
###############################################################

all_tasks = [
    ["vibrato", "straight"],
    ["belt", "breathy"],
    ["lip_trill", "spoken"],
    ["inhaled", "trill"],
    ["trillo", "vocal_fry"],
]

scenario1 = [
    ["vibrato", "straight"],
    ["belt", "breathy"],
    ["lip_trill", "spoken"],
    ["inhaled", "trill"],
    ["trillo", "vocal_fry"],
]

scenario2 = [
    ["belt", "trill"],
    ["vibrato", "inhaled"],
    ["breathy", "straight"],
    ["vocal_fry", "lip_trill"],
    ["spoken", "trillo"],
]

scenario3 = [
    ["spoken", "breathy"],
    ["straight", "inhaled"],
    ["lip_trill", "trillo"],
    ["vibrato", "vocal_fry"],
    ["trill", "belt"],
]

###############################################################
###########               COMPONENTS                ###########
###############################################################

num_classes = 10

# Data sources
train_vocalsettech_data_source = {
    "name": "VocalSetTechDataSource",
    "args": {
        "split": "train",
        "chunk_length": 5,
        "is_eval": False,
    },
}
val_vocalsettech_data_source = deepcopy(train_vocalsettech_data_source)
val_vocalsettech_data_source["args"]["split"] = "val"
val_vocalsettech_data_source["args"]["is_eval"] = True
test_vocalsettech_data_source = deepcopy(train_vocalsettech_data_source)
test_vocalsettech_data_source["args"]["split"] = "test"
test_vocalsettech_data_source["args"]["is_eval"] = True

# Metrics
classification_metrics_vocalsettech = [
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


###########                TRAINERS                 ###########

oracle_train_model_vocalsettech = {
    "name": "TorchMertClassificationModel",
    "args": {"num_classes": num_classes},
}

# Oracle
oracle_trainer_vocalsettech = deepcopy(oracle_trainer)
oracle_trainer_vocalsettech["args"]["train_model"] = oracle_train_model_vocalsettech
oracle_trainer_vocalsettech["args"][
    "train_data_source"
] = train_vocalsettech_data_source
oracle_trainer_vocalsettech["args"]["val_data_source"] = val_vocalsettech_data_source
oracle_trainer_vocalsettech["args"][
    "metrics_config"
] = classification_metrics_vocalsettech


## Finetuning
continual_learning_trainer_vocalsettech_scenario1 = deepcopy(continual_learning_trainer)
continual_learning_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_trainer_vocalsettech_scenario1
)
continual_learning_trainer_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_trainer_vocalsettech_scenario1
)
continual_learning_trainer_vocalsettech_scenario3["args"]["tasks"] = scenario3


## Replay
continual_learning_replay_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_replay_trainer
)
continual_learning_replay_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_replay_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_replay_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_replay_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_replay_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_replay_trainer_vocalsettech_scenario1
)
continual_learning_replay_trainer_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_replay_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_replay_trainer_vocalsettech_scenario1
)
continual_learning_replay_trainer_vocalsettech_scenario3["args"]["tasks"] = scenario3


## iCaRL
continual_learning_icarl_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_icarl_trainer
)
continual_learning_icarl_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_icarl_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_icarl_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_icarl_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_icarl_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_icarl_trainer_vocalsettech_scenario1
)
continual_learning_icarl_trainer_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_icarl_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_icarl_trainer_vocalsettech_scenario1
)
continual_learning_icarl_trainer_vocalsettech_scenario3["args"]["tasks"] = scenario3


## VQ
continual_learning_vq_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_vq_trainer
)
continual_learning_vq_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_vq_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_vq_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_vq_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_vq_trainer_vocalsettech_scenario1
)
continual_learning_vq_trainer_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_vq_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_vq_trainer_vocalsettech_scenario1
)
continual_learning_vq_trainer_vocalsettech_scenario3["args"]["tasks"] = scenario3


# DKVB
continual_learning_dkvb_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_dkvb_trainer
)
continual_learning_dkvb_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_dkvb_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_dkvb_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_dkvb_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_dkvb_trainer_vocalsettech_scenario1
)
continual_learning_dkvb_trainer_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_dkvb_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_dkvb_trainer_vocalsettech_scenario1
)
continual_learning_dkvb_trainer_vocalsettech_scenario3["args"]["tasks"] = scenario3


## GEM
continual_learning_gem_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_gem_trainer
)
continual_learning_gem_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_gem_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_gem_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_gem_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_gem_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_gem_trainer_vocalsettech_scenario1
)
continual_learning_gem_trainer_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_gem_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_gem_trainer_vocalsettech_scenario1
)
continual_learning_gem_trainer_vocalsettech_scenario3["args"]["tasks"] = scenario3


## EWC
continual_learning_ewc_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_ewc_trainer
)
continual_learning_ewc_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_ewc_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_ewc_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_ewc_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_ewc_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_ewc_trainer_vocalsettech_scenario1
)
continual_learning_ewc_trainer_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_ewc_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_ewc_trainer_vocalsettech_scenario1
)
continual_learning_ewc_trainer_vocalsettech_scenario3["args"]["tasks"] = scenario3


## L2P
continual_learning_l2p_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_l2p_trainer
)
continual_learning_l2p_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_l2p_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_l2p_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_l2p_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_l2p_trainer_vocalsettech_scenario1
)
continual_learning_l2p_trainer_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_l2p_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_l2p_trainer_vocalsettech_scenario1
)
continual_learning_l2p_trainer_vocalsettech_scenario3["args"]["tasks"] = scenario3


## L2Center
continual_learning_l2center_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_l2center_trainer
)
continual_learning_l2center_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_l2center_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_l2center_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_l2center_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_l2center_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_l2center_trainer_vocalsettech_scenario1
)
continual_learning_l2center_trainer_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_l2center_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_l2center_trainer_vocalsettech_scenario1
)
continual_learning_l2center_trainer_vocalsettech_scenario3["args"]["tasks"] = scenario3

## CosineCenter
continual_learning_cosinecenter_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_l2center_trainer_vocalsettech_scenario1
)
continual_learning_cosinecenter_trainer_vocalsettech_scenario1["args"][
    "train_model"
] = train_model_cosinecenter

continual_learning_cosinecenter_trainer_vocalsettech_scenario2 = deepcopy(
    continual_learning_cosinecenter_trainer_vocalsettech_scenario1
)
continual_learning_cosinecenter_trainer_vocalsettech_scenario2["args"][
    "tasks"
] = scenario2
continual_learning_cosinecenter_trainer_vocalsettech_scenario3 = deepcopy(
    continual_learning_cosinecenter_trainer_vocalsettech_scenario1
)
continual_learning_cosinecenter_trainer_vocalsettech_scenario3["args"][
    "tasks"
] = scenario3

###########               EVALUATORS                ###########

# Oracle
oracle_evaluator_vocalsettech = deepcopy(oracle_evaluator)
oracle_evaluator_vocalsettech["args"]["model"] = oracle_train_model_vocalsettech
oracle_evaluator_vocalsettech["args"]["tasks"] = all_tasks
oracle_evaluator_vocalsettech["args"]["data_source"] = test_vocalsettech_data_source
oracle_evaluator_vocalsettech["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

oracle_evaluator_vocalsettech_scenario1 = deepcopy(oracle_evaluator_vocalsettech)
oracle_evaluator_vocalsettech_scenario1["args"]["tasks"] = scenario1
oracle_evaluator_vocalsettech_scenario2 = deepcopy(oracle_evaluator_vocalsettech)
oracle_evaluator_vocalsettech_scenario2["args"]["tasks"] = scenario2
oracle_evaluator_vocalsettech_scenario3 = deepcopy(oracle_evaluator_vocalsettech)
oracle_evaluator_vocalsettech_scenario3["args"]["tasks"] = scenario3

## Finetuning
continual_learning_evaluator_vocalsettech_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_evaluator_vocalsettech_scenario2 = deepcopy(
    continual_learning_evaluator_vocalsettech_scenario1
)
continual_learning_evaluator_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_evaluator_vocalsettech_scenario3 = deepcopy(
    continual_learning_evaluator_vocalsettech_scenario1
)
continual_learning_evaluator_vocalsettech_scenario3["args"]["tasks"] = scenario3

## VQ
continual_learning_vq_evaluator_vocalsettech_scenario1 = deepcopy(
    continual_learning_evaluator_vq
)
continual_learning_vq_evaluator_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_vq_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_vq_evaluator_vocalsettech_scenario2 = deepcopy(
    continual_learning_vq_evaluator_vocalsettech_scenario1
)
continual_learning_vq_evaluator_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_vq_evaluator_vocalsettech_scenario3 = deepcopy(
    continual_learning_vq_evaluator_vocalsettech_scenario1
)
continual_learning_vq_evaluator_vocalsettech_scenario3["args"]["tasks"] = scenario3


## DKVB
continual_learning_dkvb_evaluator_vocalsettech_scenario1 = deepcopy(
    continual_learning_evaluator_dkvb
)
continual_learning_dkvb_evaluator_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_dkvb_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_dkvb_evaluator_vocalsettech_scenario2 = deepcopy(
    continual_learning_dkvb_evaluator_vocalsettech_scenario1
)
continual_learning_dkvb_evaluator_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_dkvb_evaluator_vocalsettech_scenario3 = deepcopy(
    continual_learning_dkvb_evaluator_vocalsettech_scenario1
)
continual_learning_dkvb_evaluator_vocalsettech_scenario3["args"]["tasks"] = scenario3


## L2P
continual_learning_l2p_evaluator_vocalsettech_scenario1 = deepcopy(
    continual_learning_evaluator_l2p
)
continual_learning_l2p_evaluator_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_l2p_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_l2p_evaluator_vocalsettech_scenario2 = deepcopy(
    continual_learning_l2p_evaluator_vocalsettech_scenario1
)
continual_learning_l2p_evaluator_vocalsettech_scenario2["args"]["tasks"] = scenario2
continual_learning_l2p_evaluator_vocalsettech_scenario3 = deepcopy(
    continual_learning_l2p_evaluator_vocalsettech_scenario1
)
continual_learning_l2p_evaluator_vocalsettech_scenario3["args"]["tasks"] = scenario3


## L2Center
continual_learning_l2center_evaluator_vocalsettech_scenario1 = deepcopy(
    continual_learning_evaluator_l2center
)
continual_learning_l2center_evaluator_vocalsettech_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_l2center_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_l2center_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = classification_metrics_vocalsettech

continual_learning_l2center_evaluator_vocalsettech_scenario2 = deepcopy(
    continual_learning_l2center_evaluator_vocalsettech_scenario1
)
continual_learning_l2center_evaluator_vocalsettech_scenario2["args"][
    "tasks"
] = scenario2
continual_learning_l2center_evaluator_vocalsettech_scenario3 = deepcopy(
    continual_learning_l2center_evaluator_vocalsettech_scenario1
)
continual_learning_l2center_evaluator_vocalsettech_scenario3["args"][
    "tasks"
] = scenario3

###############################################################
###########               EXPERIMENTS               ###########
###############################################################

###########                BASELINES                ###########

mert95m_base_oracle_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_base_oracle_vocalsettech_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_vocalsettech,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_vocalsettech_scenario1,
    },
}

mert95m_base_oracle_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_base_oracle_vocalsettech_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_vocalsettech,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_vocalsettech_scenario2,
    },
}

mert95m_base_oracle_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_base_oracle_vocalsettech_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_vocalsettech,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_vocalsettech_scenario3,
    },
}


###########            CONTINUAL LEARNING           ###########

# SCENARIO 1

mert95m_finetuning_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_finetuning_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario1,
    },
}

mert95m_replay_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_replay_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario1,
    },
}

mert95m_icarl_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_icarl_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario1,
    },
}

mert95m_vq_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_vq_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_vocalsettech_scenario1,
    },
}

mert95m_dkvb_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_dkvb_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_vocalsettech_scenario1,
    },
}

mert95m_gem_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_gem_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario1,
    },
}

mert95m_ewc_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_ewc_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario1,
    },
}

mert95m_l2p_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_l2p_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_vocalsettech_scenario1,
    },
}

mert95m_l2center_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_l2center_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsettech_scenario1,
    },
}

mert95m_cosinecenter_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_cosinecenter_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "CosineCenter",
    # data
    "train": {
        "trainer": continual_learning_cosinecenter_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsettech_scenario1,
    },
}


# SCENARIO 2

mert95m_finetuning_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_finetuning_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario2,
    },
}

mert95m_replay_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_replay_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario2,
    },
}

mert95m_icarl_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_icarl_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario2,
    },
}

mert95m_vq_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_vq_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_vocalsettech_scenario2,
    },
}

mert95m_dkvb_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_dkvb_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_vocalsettech_scenario2,
    },
}

mert95m_gem_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_gem_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario2,
    },
}

mert95m_ewc_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_ewc_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario2,
    },
}

mert95m_l2p_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_l2p_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_vocalsettech_scenario2,
    },
}

mert95m_l2center_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_l2center_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsettech_scenario2,
    },
}

mert95m_cosinecenter_cl_vocalsettech_scenario2 = {
    "experiment_name": "mert95m_cosinecenter_cl_vocalsettech_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "CosineCenter",
    # data
    "train": {
        "trainer": continual_learning_cosinecenter_trainer_vocalsettech_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsettech_scenario2,
    },
}


# SCENARIO 3

mert95m_finetuning_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_finetuning_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario3,
    },
}

mert95m_replay_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_replay_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario3,
    },
}

mert95m_icarl_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_icarl_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario3,
    },
}

mert95m_vq_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_vq_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_vocalsettech_scenario3,
    },
}

mert95m_dkvb_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_dkvb_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_vocalsettech_scenario3,
    },
}

mert95m_gem_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_gem_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario3,
    },
}

mert95m_ewc_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_ewc_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsettech_scenario3,
    },
}

mert95m_l2p_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_l2p_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_vocalsettech_scenario3,
    },
}

mert95m_l2center_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_l2center_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsettech_scenario3,
    },
}

mert95m_cosinecenter_cl_vocalsettech_scenario3 = {
    "experiment_name": "mert95m_cosinecenter_cl_vocalsettech_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "CosineCenter",
    # data
    "train": {
        "trainer": continual_learning_cosinecenter_trainer_vocalsettech_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_vocalsettech_scenario3,
    },
}
