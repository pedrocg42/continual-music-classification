from copy import deepcopy

from experiments.components_clmr import (
    oracle_trainer,
    oracle_evaluator,
    continual_learning_trainer,
    continual_learning_replay_trainer,
    continual_learning_ewc_trainer,
    continual_learning_gem_trainer,
    continual_learning_icarl_trainer,
    continual_learning_l2p_trainer,
    continual_learning_l2center_trainer,
    continual_learning_evaluator_l2center,
    continual_learning_evaluator_l2p,
    evaluator,
)

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
        "chunk_length": 2.6780,  # 59049 samples
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
    "name": "TorchClmrClassificationModel",
    "args": {
        "num_classes": num_classes,
        "encoder": {
            "name": "ClmrEncoder",
            "args": {
                "pretrained": True,
            },
        },
    },
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

update_trainer_gtzan = dict(
    train_data_source=train_gtzan_data_source,
    val_data_source=val_gtzan_data_source,
    metrics_config=classification_metrics_gtzan,
)

# Scenario 1

## Oracle
oracle_trainer_gtzan = deepcopy(oracle_trainer)
oracle_trainer_gtzan["args"]["train_model"] = oracle_train_model_gtzan
oracle_trainer_gtzan["args"].update(update_trainer_gtzan)

## Finetuning
continual_learning_trainer_gtzan_scenario1 = deepcopy(continual_learning_trainer)
continual_learning_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_trainer_gtzan_scenario1["args"].update(update_trainer_gtzan)

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
continual_learning_replay_trainer_gtzan_scenario1["args"].update(update_trainer_gtzan)

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
continual_learning_icarl_trainer_gtzan_scenario1["args"].update(update_trainer_gtzan)

continual_learning_icarl_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_icarl_trainer_gtzan_scenario1
)
continual_learning_icarl_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_icarl_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_icarl_trainer_gtzan_scenario1
)
continual_learning_icarl_trainer_gtzan_scenario3["args"]["tasks"] = scenario3

# GEM
continual_learning_gem_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_gem_trainer
)
continual_learning_gem_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_gem_trainer_gtzan_scenario1["args"].update(update_trainer_gtzan)

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
continual_learning_ewc_trainer_gtzan_scenario1["args"].update(update_trainer_gtzan)

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
continual_learning_l2p_trainer_gtzan_scenario1["args"].update(update_trainer_gtzan)

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
continual_learning_l2center_trainer_gtzan_scenario1["args"].update(update_trainer_gtzan)

continual_learning_l2center_trainer_gtzan_scenario2 = deepcopy(
    continual_learning_l2center_trainer_gtzan_scenario1
)
continual_learning_l2center_trainer_gtzan_scenario2["args"]["tasks"] = scenario2
continual_learning_l2center_trainer_gtzan_scenario3 = deepcopy(
    continual_learning_l2center_trainer_gtzan_scenario1
)
continual_learning_l2center_trainer_gtzan_scenario3["args"]["tasks"] = scenario3

###########               EVALUATORS                ###########

update_evaluator_gtzan = dict(
    data_source=test_gtzan_data_source,
    metrics_config=classification_metrics_gtzan,
)

# Scenario 1

## Oracle
oracle_evaluator_gtzan = deepcopy(oracle_evaluator)
oracle_evaluator_gtzan["args"]["model"] = oracle_train_model_gtzan
oracle_evaluator_gtzan["args"].update(update_evaluator_gtzan)

oracle_evaluator_gtzan_scenario1 = deepcopy(oracle_evaluator_gtzan)
oracle_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
oracle_evaluator_gtzan_scenario2 = deepcopy(oracle_evaluator_gtzan)
oracle_evaluator_gtzan_scenario2["args"]["tasks"] = scenario2
oracle_evaluator_gtzan_scenario3 = deepcopy(oracle_evaluator_gtzan)
oracle_evaluator_gtzan_scenario3["args"]["tasks"] = scenario3

## Finetuning
continual_learning_evaluator_gtzan_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_evaluator_gtzan_scenario1["args"].update(update_evaluator_gtzan)

continual_learning_evaluator_gtzan_scenario2 = deepcopy(
    continual_learning_evaluator_gtzan_scenario1
)
continual_learning_evaluator_gtzan_scenario2["args"]["tasks"] = scenario2

continual_learning_evaluator_gtzan_scenario3 = deepcopy(
    continual_learning_evaluator_gtzan_scenario1
)
continual_learning_evaluator_gtzan_scenario3["args"]["tasks"] = scenario3


## L2P
continual_learning_l2p_evaluator_gtzan_scenario1 = deepcopy(
    continual_learning_evaluator_l2p
)
continual_learning_l2p_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_evaluator_gtzan_scenario1["args"].update(update_evaluator_gtzan)

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
continual_learning_l2center_evaluator_gtzan_scenario1["args"].update(
    update_evaluator_gtzan
)

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

clmrsamplecnn_base_oracle_gtzan_scenario1 = {
    "experiment_name": "clmrsamplecnn_base_oracle_gtzan_all",
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

clmrsamplecnn_base_oracle_gtzan_scenario2 = {
    "experiment_name": "clmrsamplecnn_base_oracle_gtzan_all",
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

clmrsamplecnn_base_oracle_gtzan_scenario3 = {
    "experiment_name": "clmrsamplecnn_base_oracle_gtzan_all",
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

clmrsamplecnn_finetuning_cl_gtzan_scenario1 = {
    "experiment_name": "clmrsamplecnn_finetuning_cl_gtzan_scenario1",
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

clmrsamplecnn_replay_cl_gtzan_scenario1 = {
    "experiment_name": "clmrsamplecnn_replay_cl_gtzan_scenario1",
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

clmrsamplecnn_icarl_cl_gtzan_scenario1 = {
    "experiment_name": "clmrsamplecnn_icarl_cl_gtzan_scenario1",
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

clmrsamplecnn_gem_cl_gtzan_scenario1 = {
    "experiment_name": "clmrsamplecnn_gem_cl_gtzan_scenario1",
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

clmrsamplecnn_ewc_cl_gtzan_scenario1 = {
    "experiment_name": "clmrsamplecnn_ewc_cl_gtzan_scenario1",
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

clmrsamplecnn_l2p_cl_gtzan_scenario1 = {
    "experiment_name": "clmrsamplecnn_l2p_cl_gtzan_scenario1",
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

clmrsamplecnn_l2center_cl_gtzan_scenario1 = {
    "experiment_name": "clmrsamplecnn_l2center_cl_gtzan_scenario1",
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


# SCENARIO 2

clmrsamplecnn_finetuning_cl_gtzan_scenario2 = {
    "experiment_name": "clmrsamplecnn_finetuning_cl_gtzan_scenario2",
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

clmrsamplecnn_replay_cl_gtzan_scenario2 = {
    "experiment_name": "clmrsamplecnn_replay_cl_gtzan_scenario2",
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

clmrsamplecnn_icarl_cl_gtzan_scenario2 = {
    "experiment_name": "clmrsamplecnn_icarl_cl_gtzan_scenario2",
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

clmrsamplecnn_gem_cl_gtzan_scenario2 = {
    "experiment_name": "clmrsamplecnn_gem_cl_gtzan_scenario2",
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

clmrsamplecnn_ewc_cl_gtzan_scenario2 = {
    "experiment_name": "clmrsamplecnn_ewc_cl_gtzan_scenario2",
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

clmrsamplecnn_l2p_cl_gtzan_scenario2 = {
    "experiment_name": "clmrsamplecnn_l2p_cl_gtzan_scenario2",
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

clmrsamplecnn_l2center_cl_gtzan_scenario2 = {
    "experiment_name": "clmrsamplecnn_l2center_cl_gtzan_scenario2",
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


# SCENARIO 3

clmrsamplecnn_finetuning_cl_gtzan_scenario3 = {
    "experiment_name": "clmrsamplecnn_finetuning_cl_gtzan_scenario3",
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

clmrsamplecnn_replay_cl_gtzan_scenario3 = {
    "experiment_name": "clmrsamplecnn_replay_cl_gtzan_scenario3",
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

clmrsamplecnn_icarl_cl_gtzan_scenario3 = {
    "experiment_name": "clmrsamplecnn_icarl_cl_gtzan_scenario3",
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

clmrsamplecnn_gem_cl_gtzan_scenario3 = {
    "experiment_name": "clmrsamplecnn_gem_cl_gtzan_scenario3",
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

clmrsamplecnn_ewc_cl_gtzan_scenario3 = {
    "experiment_name": "clmrsamplecnn_ewc_cl_gtzan_scenario3",
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

clmrsamplecnn_l2p_cl_gtzan_scenario3 = {
    "experiment_name": "clmrsamplecnn_l2p_cl_gtzan_scenario3",
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

clmrsamplecnn_l2center_cl_gtzan_scenario3 = {
    "experiment_name": "clmrsamplecnn_l2center_cl_gtzan_scenario3",
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
