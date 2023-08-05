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

###############################################################
###########               COMPONENTS                ###########
###############################################################

num_classes = 20

# Data sources
train_vocalsetsingers_data_source = {
    "name": "VocalSetSingersDataSource",
    "args": {
        "split": "train",
        "chunk_length": 3,
        "num_cross_val_splits": 5,
        "is_eval": False,
    },
}
val_vocalsetsingers_data_source = deepcopy(train_vocalsetsingers_data_source)
val_vocalsetsingers_data_source["args"]["split"] = "val"
val_vocalsetsingers_data_source["args"]["is_eval"] = True
test_vocalsetsingers_data_source = deepcopy(train_vocalsetsingers_data_source)
test_vocalsetsingers_data_source["args"]["split"] = "test"
test_vocalsetsingers_data_source["args"]["is_eval"] = True

# Metrics
genre_classification_metrics_vocalsetsingers = [
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

oracle_train_model_vocalsetsingers = {
    "name": "TorchMertClassificationModel",
    "args": {"num_classes": num_classes},
}

# Scenario 1
oracle_trainer_vocalsetsingers = deepcopy(oracle_trainer)
oracle_trainer_vocalsetsingers["args"][
    "train_model"
] = oracle_train_model_vocalsetsingers
oracle_trainer_vocalsetsingers["args"][
    "train_data_source"
] = train_vocalsetsingers_data_source
oracle_trainer_vocalsetsingers["args"][
    "val_data_source"
] = val_vocalsetsingers_data_source
oracle_trainer_vocalsetsingers["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_trainer_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_trainer
)
continual_learning_trainer_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_trainer_vocalsetsingers_scenario1["args"][
    "train_data_source"
] = train_vocalsetsingers_data_source
continual_learning_trainer_vocalsetsingers_scenario1["args"][
    "val_data_source"
] = val_vocalsetsingers_data_source
continual_learning_trainer_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_replay_trainer_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_replay_trainer
)
continual_learning_replay_trainer_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_replay_trainer_vocalsetsingers_scenario1["args"][
    "train_data_source"
] = train_vocalsetsingers_data_source
continual_learning_replay_trainer_vocalsetsingers_scenario1["args"][
    "val_data_source"
] = val_vocalsetsingers_data_source
continual_learning_replay_trainer_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_icarl_trainer_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_icarl_trainer
)
continual_learning_icarl_trainer_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_icarl_trainer_vocalsetsingers_scenario1["args"][
    "train_data_source"
] = train_vocalsetsingers_data_source
continual_learning_icarl_trainer_vocalsetsingers_scenario1["args"][
    "val_data_source"
] = val_vocalsetsingers_data_source
continual_learning_icarl_trainer_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_vq_trainer_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_vq_trainer
)
continual_learning_vq_trainer_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_trainer_vocalsetsingers_scenario1["args"][
    "train_data_source"
] = train_vocalsetsingers_data_source
continual_learning_vq_trainer_vocalsetsingers_scenario1["args"][
    "val_data_source"
] = val_vocalsetsingers_data_source
continual_learning_vq_trainer_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_dkvb_trainer_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_dkvb_trainer
)
continual_learning_dkvb_trainer_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_trainer_vocalsetsingers_scenario1["args"][
    "train_data_source"
] = train_vocalsetsingers_data_source
continual_learning_dkvb_trainer_vocalsetsingers_scenario1["args"][
    "val_data_source"
] = val_vocalsetsingers_data_source
continual_learning_dkvb_trainer_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_gem_trainer_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_gem_trainer
)
continual_learning_gem_trainer_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_gem_trainer_vocalsetsingers_scenario1["args"][
    "train_data_source"
] = train_vocalsetsingers_data_source
continual_learning_gem_trainer_vocalsetsingers_scenario1["args"][
    "val_data_source"
] = val_vocalsetsingers_data_source
continual_learning_gem_trainer_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_ewc_trainer_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_ewc_trainer
)
continual_learning_ewc_trainer_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_ewc_trainer_vocalsetsingers_scenario1["args"][
    "train_data_source"
] = train_vocalsetsingers_data_source
continual_learning_ewc_trainer_vocalsetsingers_scenario1["args"][
    "val_data_source"
] = val_vocalsetsingers_data_source
continual_learning_ewc_trainer_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_l2p_trainer_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_l2p_trainer
)
continual_learning_l2p_trainer_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_trainer_vocalsetsingers_scenario1["args"][
    "train_data_source"
] = train_vocalsetsingers_data_source
continual_learning_l2p_trainer_vocalsetsingers_scenario1["args"][
    "val_data_source"
] = val_vocalsetsingers_data_source
continual_learning_l2p_trainer_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


###########               EVALUATORS                ###########

# Scenario 1
oracle_evaluator_vocalsetsingers = deepcopy(oracle_evaluator)
oracle_evaluator_vocalsetsingers["args"]["model"] = oracle_train_model_vocalsetsingers
oracle_evaluator_vocalsetsingers["args"]["tasks"] = scenario1
oracle_evaluator_vocalsetsingers["args"][
    "data_source"
] = test_vocalsetsingers_data_source
oracle_evaluator_vocalsetsingers["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers

continual_learning_evaluator_vocalsetsingers_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_evaluator_vocalsetsingers_scenario1["args"][
    "data_source"
] = test_vocalsetsingers_data_source
continual_learning_evaluator_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_vq_evaluator_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_evaluator_vq
)
continual_learning_vq_evaluator_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_evaluator_vocalsetsingers_scenario1["args"][
    "data_source"
] = test_vocalsetsingers_data_source
continual_learning_vq_evaluator_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_dkvb_evaluator_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_evaluator_dkvb
)
continual_learning_dkvb_evaluator_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_evaluator_vocalsetsingers_scenario1["args"][
    "data_source"
] = test_vocalsetsingers_data_source
continual_learning_dkvb_evaluator_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


continual_learning_l2p_evaluator_vocalsetsingers_scenario1 = deepcopy(
    continual_learning_evaluator_l2p
)
continual_learning_l2p_evaluator_vocalsetsingers_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_evaluator_vocalsetsingers_scenario1["args"][
    "data_source"
] = test_vocalsetsingers_data_source
continual_learning_l2p_evaluator_vocalsetsingers_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsetsingers


###############################################################
###########               EXPERIMENTS               ###########
###############################################################

###########                BASELINES                ###########

mert95m_base_oracle_vocalsetsingers_all = {
    "experiment_name": "mert95m_base_oracle_vocalsetsingers_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": oracle_trainer_vocalsetsingers,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_vocalsetsingers,
    },
}


###########            CONTINUAL LEARNING           ###########

# SCENARIO 1

mert95m_finetuning_cl_vocalsetsingers_scenario1 = {
    "experiment_name": "mert95m_finetuning_cl_vocalsetsingers_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_trainer_vocalsetsingers_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsingers_scenario1,
    },
}

mert95m_replay_cl_vocalsetsingers_scenario1 = {
    "experiment_name": "mert95m_replay_cl_vocalsetsingers_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_vocalsetsingers_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsingers_scenario1,
    },
}

mert95m_icarl_cl_vocalsetsingers_scenario1 = {
    "experiment_name": "mert95m_icarl_cl_vocalsetsingers_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "iCaRL",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_icarl_trainer_vocalsetsingers_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsingers_scenario1,
    },
}

mert95m_vq_cl_vocalsetsingers_scenario1 = {
    "experiment_name": "mert95m_vq_cl_vocalsetsingers_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_vocalsetsingers_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_vocalsetsingers_scenario1,
    },
}

mert95m_dkvb_cl_vocalsetsingers_scenario1 = {
    "experiment_name": "mert95m_dkvb_cl_vocalsetsingers_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_vocalsetsingers_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_vocalsetsingers_scenario1,
    },
}

mert95m_gem_cl_vocalsetsingers_scenario1 = {
    "experiment_name": "mert95m_gem_cl_vocalsetsingers_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_vocalsetsingers_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsingers_scenario1,
    },
}

mert95m_ewc_cl_vocalsetsingers_scenario1 = {
    "experiment_name": "mert95m_ewc_cl_vocalsetsingers_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_vocalsetsingers_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_vocalsetsingers_scenario1,
    },
}

mert95m_l2p_cl_vocalsetsingers_scenario1 = {
    "experiment_name": "mert95m_l2p_cl_vocalsetsingers_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_vocalsetsingers_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_vocalsetsingers_scenario1,
    },
}
