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
    mert_data_transform_nsynth,
)

###############################################################
###########                SCENARIOS                ###########
###############################################################

all_tasks = [
    ["bass", "brass"],
    ["flute", "guitar"],
    ["keyboard", "mallet"],
    ["organ", "reed"],
    ["string", "synth_lead", "vocal"],
]

scenario1 = [
    ["bass", "brass"],
    ["flute", "guitar"],
    ["keyboard", "mallet"],
    ["organ", "reed"],
    ["string", "synth_lead", "vocal"],
]

scenario2 = [
    ["keyboard", "synth_lead"],
    ["mallet", "guitar"],
    ["reed", "string", "bass"],
    ["vocal", "brass"],
    ["organ", "flute"],
]

scenario3 = [
    ["guitar", "synth_lead", "organ"],
    ["string", "brass"],
    ["mallet", "reed"],
    ["keyboard", "bass"],
    ["flute", "vocal"],
]


###############################################################
###########               COMPONENTS                ###########
###############################################################

num_classes = 11
early_stopping_patience = 3

# Data sources
train_nsynthinstrument_data_source = {
    "name": "NSynthInstrumentTechDataSource",
    "args": {
        "split": "train",
        "num_items_per_class": 5000,
        "chunk_length": 4,
        "is_eval": False,
    },
}
val_nsynthinstrument_data_source = deepcopy(train_nsynthinstrument_data_source)
val_nsynthinstrument_data_source["args"]["split"] = "val"
val_nsynthinstrument_data_source["args"]["is_eval"] = True
test_nsynthinstrument_data_source = deepcopy(train_nsynthinstrument_data_source)
test_nsynthinstrument_data_source["args"]["split"] = "test"
test_nsynthinstrument_data_source["args"]["is_eval"] = True

# Metrics
classification_metrics_nsynthinstrument = [
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

oracle_train_model_nsynthinstrument = {
    "name": "TorchMertClassificationModel",
    "args": {"num_classes": num_classes},
}

update_trainer_nsynthinstrument = dict(
    train_data_source=train_nsynthinstrument_data_source,
    val_data_source=val_nsynthinstrument_data_source,
    metrics_config=classification_metrics_nsynthinstrument,
    train_data_transform=mert_data_transform_nsynth,
    val_data_transform=mert_data_transform_nsynth,
    early_stopping_patience=early_stopping_patience,
)

# Scenario 1
oracle_trainer_nsynthinstrument = deepcopy(oracle_trainer)
oracle_trainer_nsynthinstrument["args"][
    "train_model"
] = oracle_train_model_nsynthinstrument
oracle_trainer_nsynthinstrument["args"].update(update_trainer_nsynthinstrument)


# Finetuning
continual_learning_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_trainer
)
continual_learning_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_trainer_nsynthinstrument_scenario1["args"].update(
    update_trainer_nsynthinstrument
)


continual_learning_trainer_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_trainer_nsynthinstrument_scenario1
)
continual_learning_trainer_nsynthinstrument_scenario2["args"]["tasks"] = scenario2

continual_learning_trainer_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_trainer_nsynthinstrument_scenario1
)
continual_learning_trainer_nsynthinstrument_scenario3["args"]["tasks"] = scenario3


## Replay
continual_learning_replay_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_replay_trainer
)
continual_learning_replay_trainer_nsynthinstrument_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_replay_trainer_nsynthinstrument_scenario1["args"].update(
    update_trainer_nsynthinstrument
)


continual_learning_replay_trainer_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_replay_trainer_nsynthinstrument_scenario1
)
continual_learning_replay_trainer_nsynthinstrument_scenario2["args"][
    "tasks"
] = scenario2

continual_learning_replay_trainer_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_replay_trainer_nsynthinstrument_scenario1
)
continual_learning_replay_trainer_nsynthinstrument_scenario3["args"][
    "tasks"
] = scenario3


## iCaRL
continual_learning_icarl_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_icarl_trainer
)
continual_learning_icarl_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_icarl_trainer_nsynthinstrument_scenario1["args"].update(
    update_trainer_nsynthinstrument
)

continual_learning_icarl_trainer_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_icarl_trainer_nsynthinstrument_scenario1
)
continual_learning_icarl_trainer_nsynthinstrument_scenario2["args"]["tasks"] = scenario2

continual_learning_icarl_trainer_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_icarl_trainer_nsynthinstrument_scenario1
)
continual_learning_icarl_trainer_nsynthinstrument_scenario3["args"]["tasks"] = scenario3

## GEM
continual_learning_gem_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_gem_trainer
)
continual_learning_gem_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_gem_trainer_nsynthinstrument_scenario1["args"].update(
    update_trainer_nsynthinstrument
)


continual_learning_gem_trainer_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_gem_trainer_nsynthinstrument_scenario1
)
continual_learning_gem_trainer_nsynthinstrument_scenario2["args"]["tasks"] = scenario2

continual_learning_gem_trainer_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_gem_trainer_nsynthinstrument_scenario1
)
continual_learning_gem_trainer_nsynthinstrument_scenario3["args"]["tasks"] = scenario3


## EWC
continual_learning_ewc_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_ewc_trainer
)
continual_learning_ewc_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_ewc_trainer_nsynthinstrument_scenario1["args"].update(
    update_trainer_nsynthinstrument
)

continual_learning_ewc_trainer_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_ewc_trainer_nsynthinstrument_scenario1
)
continual_learning_ewc_trainer_nsynthinstrument_scenario2["args"]["tasks"] = scenario2

continual_learning_ewc_trainer_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_ewc_trainer_nsynthinstrument_scenario1
)
continual_learning_ewc_trainer_nsynthinstrument_scenario3["args"]["tasks"] = scenario3


## L2P
continual_learning_l2p_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_l2p_trainer
)
continual_learning_l2p_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_trainer_nsynthinstrument_scenario1["args"].update(
    update_trainer_nsynthinstrument
)


continual_learning_l2p_trainer_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_l2p_trainer_nsynthinstrument_scenario1
)
continual_learning_l2p_trainer_nsynthinstrument_scenario2["args"]["tasks"] = scenario2

continual_learning_l2p_trainer_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_l2p_trainer_nsynthinstrument_scenario1
)
continual_learning_l2p_trainer_nsynthinstrument_scenario3["args"]["tasks"] = scenario3

## L2Center
continual_learning_l2center_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_l2center_trainer
)
continual_learning_l2center_trainer_nsynthinstrument_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_l2center_trainer_nsynthinstrument_scenario1["args"].update(
    update_trainer_nsynthinstrument
)

continual_learning_l2center_trainer_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_l2center_trainer_nsynthinstrument_scenario1
)
continual_learning_l2center_trainer_nsynthinstrument_scenario2["args"][
    "tasks"
] = scenario2

continual_learning_l2center_trainer_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_l2center_trainer_nsynthinstrument_scenario1
)
continual_learning_l2center_trainer_nsynthinstrument_scenario3["args"][
    "tasks"
] = scenario3


###########               EVALUATORS                ###########

update_evaluator_nsynthinstrument = dict(
    data_source=test_nsynthinstrument_data_source,
    metrics_config=classification_metrics_nsynthinstrument,
    data_transform=mert_data_transform_nsynth,
)

# Oracle
oracle_evaluator_nsynthinstrument = deepcopy(oracle_evaluator)
oracle_evaluator_nsynthinstrument["args"]["model"] = oracle_train_model_nsynthinstrument
oracle_evaluator_nsynthinstrument["args"]["tasks"] = all_tasks
oracle_evaluator_nsynthinstrument["args"].update(update_evaluator_nsynthinstrument)

oracle_evaluator_nsynthinstrument_scenario1 = deepcopy(
    oracle_evaluator_nsynthinstrument
)
oracle_evaluator_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
oracle_evaluator_nsynthinstrument_scenario2 = deepcopy(
    oracle_evaluator_nsynthinstrument
)
oracle_evaluator_nsynthinstrument_scenario2["args"]["tasks"] = scenario2
oracle_evaluator_nsynthinstrument_scenario3 = deepcopy(
    oracle_evaluator_nsynthinstrument
)
oracle_evaluator_nsynthinstrument_scenario3["args"]["tasks"] = scenario3

## Finetuning
continual_learning_evaluator_nsynthinstrument_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_evaluator_nsynthinstrument_scenario1["args"].update(
    update_evaluator_nsynthinstrument
)

continual_learning_evaluator_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_evaluator_nsynthinstrument_scenario1
)
continual_learning_evaluator_nsynthinstrument_scenario2["args"]["tasks"] = scenario2
continual_learning_evaluator_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_evaluator_nsynthinstrument_scenario1
)
continual_learning_evaluator_nsynthinstrument_scenario3["args"]["tasks"] = scenario3


## L2P
continual_learning_l2p_evaluator_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_evaluator_l2p
)
continual_learning_l2p_evaluator_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_evaluator_nsynthinstrument_scenario1["args"].update(
    update_evaluator_nsynthinstrument
)

continual_learning_l2p_evaluator_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_l2p_evaluator_nsynthinstrument_scenario1
)
continual_learning_l2p_evaluator_nsynthinstrument_scenario2["args"]["tasks"] = scenario2
continual_learning_l2p_evaluator_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_l2p_evaluator_nsynthinstrument_scenario1
)
continual_learning_l2p_evaluator_nsynthinstrument_scenario3["args"]["tasks"] = scenario3


# L2Center
continual_learning_l2center_evaluator_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_evaluator_l2center
)
continual_learning_l2center_evaluator_nsynthinstrument_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_l2center_evaluator_nsynthinstrument_scenario1["args"].update(
    update_evaluator_nsynthinstrument
)

continual_learning_l2center_evaluator_nsynthinstrument_scenario2 = deepcopy(
    continual_learning_l2center_evaluator_nsynthinstrument_scenario1
)
continual_learning_l2center_evaluator_nsynthinstrument_scenario2["args"][
    "tasks"
] = scenario2
continual_learning_l2center_evaluator_nsynthinstrument_scenario3 = deepcopy(
    continual_learning_l2center_evaluator_nsynthinstrument_scenario1
)
continual_learning_l2center_evaluator_nsynthinstrument_scenario3["args"][
    "tasks"
] = scenario3


###############################################################
###########               EXPERIMENTS               ###########
###############################################################

###########                BASELINES                ###########

clmrsamplecnnxl_base_oracle_nsynthinstrument_scenario1 = {
    "experiment_name": "clmrsamplecnnxl_base_oracle_nsynthinstrument_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_nsynthinstrument,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_nsynthinstrument_scenario1,
    },
}

clmrsamplecnnxl_base_oracle_nsynthinstrument_scenario2 = {
    "experiment_name": "clmrsamplecnnxl_base_oracle_nsynthinstrument_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_nsynthinstrument,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_nsynthinstrument_scenario2,
    },
}

clmrsamplecnnxl_base_oracle_nsynthinstrument_scenario3 = {
    "experiment_name": "clmrsamplecnnxl_base_oracle_nsynthinstrument_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_nsynthinstrument,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_nsynthinstrument_scenario3,
    },
}

###########            CONTINUAL LEARNING           ###########

# SCENARIO 1

clmrsamplecnnxl_finetuning_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "clmrsamplecnnxl_finetuning_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario1,
    },
}

clmrsamplecnnxl_replay_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "clmrsamplecnnxl_replay_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario1,
    },
}

clmrsamplecnnxl_icarl_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "clmrsamplecnnxl_icarl_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario1,
    },
}

clmrsamplecnnxl_gem_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "clmrsamplecnnxl_gem_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario1,
    },
}

clmrsamplecnnxl_ewc_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "clmrsamplecnnxl_ewc_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario1,
    },
}

clmrsamplecnnxl_l2p_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "clmrsamplecnnxl_l2p_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_nsynthinstrument_scenario1,
    },
}

clmrsamplecnnxl_l2center_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "clmrsamplecnnxl_l2center_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_nsynthinstrument_scenario1,
    },
}


# SCENARIO 2

clmrsamplecnnxl_finetuning_cl_nsynthinstrument_scenario2 = {
    "experiment_name": "clmrsamplecnnxl_finetuning_cl_nsynthinstrument_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_nsynthinstrument_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario2,
    },
}

clmrsamplecnnxl_replay_cl_nsynthinstrument_scenario2 = {
    "experiment_name": "clmrsamplecnnxl_replay_cl_nsynthinstrument_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_nsynthinstrument_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario2,
    },
}

clmrsamplecnnxl_icarl_cl_nsynthinstrument_scenario2 = {
    "experiment_name": "clmrsamplecnnxl_icarl_cl_nsynthinstrument_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_nsynthinstrument_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario2,
    },
}


clmrsamplecnnxl_gem_cl_nsynthinstrument_scenario2 = {
    "experiment_name": "clmrsamplecnnxl_gem_cl_nsynthinstrument_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_nsynthinstrument_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario2,
    },
}

clmrsamplecnnxl_ewc_cl_nsynthinstrument_scenario2 = {
    "experiment_name": "clmrsamplecnnxl_ewc_cl_nsynthinstrument_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_nsynthinstrument_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario2,
    },
}

clmrsamplecnnxl_l2p_cl_nsynthinstrument_scenario2 = {
    "experiment_name": "clmrsamplecnnxl_l2p_cl_nsynthinstrument_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_nsynthinstrument_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_nsynthinstrument_scenario2,
    },
}

clmrsamplecnnxl_l2center_cl_nsynthinstrument_scenario2 = {
    "experiment_name": "clmrsamplecnnxl_l2center_cl_nsynthinstrument_scenario2",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_nsynthinstrument_scenario2,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_nsynthinstrument_scenario2,
    },
}


# SCENARIO 3

clmrsamplecnnxl_finetuning_cl_nsynthinstrument_scenario3 = {
    "experiment_name": "clmrsamplecnnxl_finetuning_cl_nsynthinstrument_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    # data
    "train": {
        "trainer": continual_learning_trainer_nsynthinstrument_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario3,
    },
}

clmrsamplecnnxl_replay_cl_nsynthinstrument_scenario3 = {
    "experiment_name": "clmrsamplecnnxl_replay_cl_nsynthinstrument_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_nsynthinstrument_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario3,
    },
}

clmrsamplecnnxl_icarl_cl_nsynthinstrument_scenario3 = {
    "experiment_name": "clmrsamplecnnxl_icarl_cl_nsynthinstrument_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_nsynthinstrument_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario3,
    },
}


clmrsamplecnnxl_gem_cl_nsynthinstrument_scenario3 = {
    "experiment_name": "clmrsamplecnnxl_gem_cl_nsynthinstrument_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_nsynthinstrument_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario3,
    },
}

clmrsamplecnnxl_ewc_cl_nsynthinstrument_scenario3 = {
    "experiment_name": "clmrsamplecnnxl_ewc_cl_nsynthinstrument_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_nsynthinstrument_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_nsynthinstrument_scenario3,
    },
}

clmrsamplecnnxl_l2p_cl_nsynthinstrument_scenario3 = {
    "experiment_name": "clmrsamplecnnxl_l2p_cl_nsynthinstrument_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_nsynthinstrument_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_nsynthinstrument_scenario3,
    },
}

clmrsamplecnnxl_l2center_cl_nsynthinstrument_scenario3 = {
    "experiment_name": "clmrsamplecnnxl_l2center_cl_nsynthinstrument_scenario3",
    "experiment_type": "CL",
    "experiment_subtype": "L2Center",
    # data
    "train": {
        "trainer": continual_learning_l2center_trainer_nsynthinstrument_scenario3,
    },
    "evaluate": {
        "evaluator": continual_learning_l2center_evaluator_nsynthinstrument_scenario3,
    },
}
