from copy import deepcopy

from experiments.components import *

###############################################################
###########                SCENARIOS                ###########
###############################################################

all_tasks = [
    ["blues", "classical"],
    ["country", "disco"],
    ["hiphop", "jazz"],
    ["metal", "pop"],
    ["reggae", "rock"],
]

scenario1 = [
    ["metal", "pop"],
    ["country", "disco"],
    ["reggae", "rock"],
    ["blues", "classical"],
    ["hiphop", "jazz"],
]

scenario2 = [
    ["jazz", "rock"],
    ["classical", "hiphop"],
    ["reggae", "country"],
    ["metal", "blues"],
    ["pop", "disco"],
]

scenario3 = [
    ["hiphop", "metal"],
    ["reggae", "pop"],
    ["classical", "rock"],
    ["disco", "blues"],
    ["country", "jazz"],
]

###############################################################
###########               COMPONENTS                ###########
###############################################################

num_classes = 10

# Data sources
train_gtzan_data_source = {
    "name": "GtzanDataSource",
    "args": {
        "split": "train",
        "chunk_length": 5,
        "is_eval": False,
    },
}
val_gtzan_data_source = deepcopy(train_gtzan_data_source)
val_gtzan_data_source["args"]["split"] = "val"
val_gtzan_data_source["args"]["is_eval"] = True
test_gtzan_data_source = deepcopy(train_gtzan_data_source)
test_gtzan_data_source["args"]["split"] = "test"
test_gtzan_data_source["args"]["is_eval"] = True

oracle_train_model_gtzan = {
    "name": "TorchMertClassificationModel",
    "args": {"num_classes": num_classes},
}

# Metrics
classification_metrics_gtzan = [
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

# Scenario 1

## Oracle
oracle_trainer_gtzan = deepcopy(oracle_trainer)
oracle_trainer_gtzan["args"]["train_model"] = oracle_train_model_gtzan
oracle_trainer_gtzan["args"]["train_data_source"] = train_gtzan_data_source
oracle_trainer_gtzan["args"]["val_data_source"] = train_gtzan_data_source
oracle_trainer_gtzan["args"]["metrics_config"] = classification_metrics_gtzan

## Finetuning
continual_learning_trainer_gtzan_scenario1 = deepcopy(continual_learning_trainer)
continual_learning_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_trainer_gtzan_scenario1
)
continual_learning_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_trainer_gtzan_scenario1
)
continual_learning_trainer_gtzan_scenario3["args"]["tasks"] = scenario3

## Replay
continual_learning_replay_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_replay_trainer
)
continual_learning_replay_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_replay_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_replay_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_replay_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan


continual_learning_replay_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_replay_trainer_gtzan_scenario1
)
continual_learning_replay_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_replay_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_replay_trainer_gtzan_scenario1
)
continual_learning_replay_trainer_gtzan_scenario3["args"]["tasks"] = scenario3


# iCaRL
continual_learning_icarl_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_icarl_trainer
)
continual_learning_icarl_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_icarl_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_icarl_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_icarl_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan


continual_learning_icarl_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_icarl_trainer_gtzan_scenario1
)
continual_learning_icarl_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_icarl_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_icarl_trainer_gtzan_scenario1
)
continual_learning_icarl_trainer_gtzan_scenario3["args"]["tasks"] = scenario3

# VQ
continual_learning_vq_trainer_gtzan_scenario1 = deepcopy(continual_learning_vq_trainer)
continual_learning_vq_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_vq_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_vq_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_vq_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_vq_trainer_gtzan_scenario1
)
continual_learning_vq_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_vq_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_vq_trainer_gtzan_scenario1
)
continual_learning_vq_trainer_gtzan_scenario3["args"]["tasks"] = scenario3


# DKVB
continual_learning_dkvb_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_dkvb_trainer
)
continual_learning_dkvb_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_dkvb_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_dkvb_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_dkvb_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_dkvb_trainer_gtzan_scenario1
)
continual_learning_dkvb_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_dkvb_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_dkvb_trainer_gtzan_scenario1
)
continual_learning_dkvb_trainer_gtzan_scenario3["args"]["tasks"] = scenario3


# GEM
continual_learning_gem_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_gem_trainer
)
continual_learning_gem_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_gem_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_gem_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_gem_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_gem_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_gem_trainer_gtzan_scenario1
)
continual_learning_gem_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_gem_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_gem_trainer_gtzan_scenario1
)
continual_learning_gem_trainer_gtzan_scenario3["args"]["tasks"] = scenario3


# EWC
continual_learning_ewc_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_ewc_trainer
)
continual_learning_ewc_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_ewc_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_ewc_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_ewc_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_ewc_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_ewc_trainer_gtzan_scenario1
)
continual_learning_ewc_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_ewc_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_ewc_trainer_gtzan_scenario1
)
continual_learning_ewc_trainer_gtzan_scenario3["args"]["tasks"] = scenario3


# L2P
continual_learning_l2p_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_l2p_trainer
)
continual_learning_l2p_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_l2p_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_l2p_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan


continual_learning_l2p_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_l2p_trainer_gtzan_scenario1
)
continual_learning_l2p_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_l2p_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_l2p_trainer_gtzan_scenario1
)
continual_learning_l2p_trainer_gtzan_scenario3["args"]["tasks"] = scenario3


# L2Center
continual_learning_l2center_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_l2center_trainer
)
continual_learning_l2center_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_l2center_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_l2center_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_l2center_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_l2center_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_l2center_trainer_gtzan_scenario1
)
continual_learning_l2center_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_l2center_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_l2center_trainer_gtzan_scenario1
)
continual_learning_l2center_trainer_gtzan_scenario3["args"]["tasks"] = scenario3

## CosineCenter
continual_learning_cosinecenter_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_l2center_trainer_gtzan_scenario1
)
continual_learning_cosinecenter_trainer_gtzan_scenario1["args"][
    "train_model"
] = train_model_cosinecenter

continual_learning_cosinecenter_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_cosinecenter_trainer_gtzan_scenario1
)
continual_learning_cosinecenter_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_cosinecenter_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_cosinecenter_trainer_gtzan_scenario1
)
continual_learning_cosinecenter_trainer_gtzan_scenario3["args"]["tasks"] = scenario3

###########               EVALUATORS                ###########

# Scenario 1

## Oracle
oracle_evaluator_gtzan = deepcopy(oracle_evaluator)
oracle_evaluator_gtzan["args"]["model"] = oracle_train_model_gtzan
oracle_evaluator_gtzan["args"]["tasks"] = all_tasks
oracle_evaluator_gtzan["args"]["data_source"] = test_gtzan_data_source
oracle_evaluator_gtzan["args"]["metrics_config"] = classification_metrics_gtzan

oracle_evaluator_gtzan_scenario1 = deepcopy(oracle_evaluator_gtzan)
oracle_evaluator_gtzan_scenario1 = scenario1
oracle_evaluator_gtzan_scenario2 = deepcopy(oracle_evaluator_gtzan)
oracle_evaluator_gtzan_scenario2 = scenario2
oracle_evaluator_gtzan_scenario3 = deepcopy(oracle_evaluator_gtzan)
oracle_evaluator_gtzan_scenario3 = scenario3

## Finetuning
continual_learning_evaluator_gtzan_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_evaluator_gtzan_scenario2 = deepcopy(
    continual_learning_evaluator_gtzan_scenario1
)
continual_learning_evaluator_gtzan_scenario2["args"]["tasks"] = scenario2

continual_learning_evaluator_gtzan_scenario3 = deepcopy(
    continual_learning_evaluator_gtzan_scenario1
)
continual_learning_evaluator_gtzan_scenario3["args"]["tasks"] = scenario3

## VQ
continual_learning_vq_evaluator_gtzan_scenario1 = deepcopy(
    continual_learning_evaluator_vq
)
continual_learning_vq_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_vq_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_vq_evaluator_gtzan_scenario2 = deepcopy(
    continual_learning_vq_evaluator_gtzan_scenario1
)
continual_learning_vq_evaluator_gtzan_scenario2["args"]["tasks"] = scenario2

continual_learning_vq_evaluator_gtzan_scenario3 = deepcopy(
    continual_learning_vq_evaluator_gtzan_scenario1
)
continual_learning_vq_evaluator_gtzan_scenario3["args"]["tasks"] = scenario3


## DKVB
continual_learning_dkvb_evaluator_gtzan_scenario1 = deepcopy(
    continual_learning_evaluator_dkvb
)
continual_learning_dkvb_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_dkvb_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_dkvb_evaluator_gtzan_scenario2 = deepcopy(
    continual_learning_dkvb_evaluator_gtzan_scenario1
)
continual_learning_dkvb_evaluator_gtzan_scenario2["args"]["tasks"] = scenario2

continual_learning_dkvb_evaluator_gtzan_scenario3 = deepcopy(
    continual_learning_dkvb_evaluator_gtzan_scenario1
)
continual_learning_dkvb_evaluator_gtzan_scenario3["args"]["tasks"] = scenario3


## L2P
continual_learning_l2p_evaluator_gtzan_scenario1 = deepcopy(
    continual_learning_evaluator_l2p
)
continual_learning_l2p_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_l2p_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_l2p_evaluator_gtzan_scenario2 = deepcopy(
    continual_learning_l2p_evaluator_gtzan_scenario1
)
continual_learning_l2p_evaluator_gtzan_scenario2["args"]["tasks"] = scenario2

continual_learning_l2p_evaluator_gtzan_scenario3 = deepcopy(
    continual_learning_l2p_evaluator_gtzan_scenario1
)
continual_learning_l2p_evaluator_gtzan_scenario3["args"]["tasks"] = scenario3

# L2Center
continual_learning_l2center_evaluator_gtzan_scenario1 = deepcopy(
    continual_learning_evaluator_l2center
)
continual_learning_l2center_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_l2center_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_l2center_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = classification_metrics_gtzan

continual_learning_l2center_evaluator_gtzan_scenario2 = deepcopy(
    continual_learning_l2center_evaluator_gtzan_scenario1
)
continual_learning_l2center_evaluator_gtzan_scenario2["args"]["tasks"] = scenario2

continual_learning_l2center_evaluator_gtzan_scenario3 = deepcopy(
    continual_learning_l2center_evaluator_gtzan_scenario1
)
continual_learning_l2center_evaluator_gtzan_scenario3["args"]["tasks"] = scenario3


###############################################################
###########               EXPERIMENTS               ###########
###############################################################

###########                BASELINES                ###########

mert95m_base_oracle_gtzan_scenario1 = {
    "experiment_name": "mert95m_base_oracle_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_gtzan,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_gtzan_scenario1,
    },
}

mert95m_base_oracle_gtzan_scenario2 = {
    "experiment_name": "mert95m_base_oracle_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_gtzan,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_gtzan_scenario2,
    },
}

mert95m_base_oracle_gtzan_scenario3 = {
    "experiment_name": "mert95m_base_oracle_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_gtzan,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_gtzan_scenario3,
    },
}


###########            CONTINUAL LEARNING           ###########

# SCENARIO 1

mert95m_finetuning_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_finetuning_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario1,
    },
}

mert95m_replay_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_replay_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario1,
    },
}

mert95m_icarl_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_icarl_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario1,
    },
}

mert95m_vq_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_vq_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_gtzan_scenario1,
    },
}

mert95m_dkvb_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_dkvb_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_gtzan_scenario1,
    },
}

mert95m_gem_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_gem_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario1,
    },
}

mert95m_ewc_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_ewc_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario1,
    },
}

mert95m_l2p_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_l2p_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_gtzan_scenario1,
    },
}

mert95m_l2center_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_l2center_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_gtzan_scenario1,
    },
}

mert95m_cosinecenter_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_cosinecenter_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "CosineCenter",
    # data
    "train": {
        "trainer": continual_learning_cosinecenter_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_gtzan_scenario1,
    },
}


# SCENARIO 2

mert95m_finetuning_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_finetuning_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario2,
    },
}

mert95m_replay_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_replay_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario2,
    },
}

mert95m_icarl_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_icarl_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario2,
    },
}

mert95m_vq_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_vq_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_gtzan_scenario2,
    },
}

mert95m_dkvb_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_dkvb_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_gtzan_scenario2,
    },
}

mert95m_gem_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_gem_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario2,
    },
}

mert95m_ewc_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_ewc_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario2,
    },
}

mert95m_l2p_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_l2p_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_gtzan_scenario2,
    },
}

mert95m_l2center_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_l2center_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_gtzan_scenario2,
    },
}

mert95m_cosinecenter_cl_gtzan_scenario2 = {
    "experiment_name": "mert95m_cosinecenter_cl_gtzan_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "CosineCenter",
    # data
    "train": {
        "trainer": continual_learning_cosinecenter_trainer_gtzan_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_gtzan_scenario2,
    },
}


# SCENARIO 3

mert95m_finetuning_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_finetuning_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario3,
    },
}

mert95m_replay_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_replay_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario3,
    },
}

mert95m_icarl_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_icarl_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario3,
    },
}

mert95m_vq_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_vq_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_gtzan_scenario3,
    },
}

mert95m_dkvb_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_dkvb_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_gtzan_scenario3,
    },
}

mert95m_gem_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_gem_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario3,
    },
}

mert95m_ewc_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_ewc_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_gtzan_scenario3,
    },
}

mert95m_l2p_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_l2p_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_gtzan_scenario3,
    },
}

mert95m_l2center_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_l2center_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_gtzan_scenario3,
    },
}

mert95m_cosinecenter_cl_gtzan_scenario3 = {
    "experiment_name": "mert95m_cosinecenter_cl_gtzan_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "CosineCenter",
    # data
    "train": {
        "trainer": continual_learning_cosinecenter_trainer_gtzan_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_gtzan_scenario3,
    },
}
