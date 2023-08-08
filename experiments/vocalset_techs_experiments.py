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


###############################################################
###########               COMPONENTS                ###########
###############################################################

num_classes = 10

# Data sources
train_vocalsettech_data_source = {
    "name": "VocalSetTechDataSource",
    "args": {
        "split": "train",
        "chunk_length": 3,
        "num_cross_val_splits": 5,
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
genre_classification_metrics_vocalsettech = [
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

# Scenario 1
oracle_trainer_vocalsettech = deepcopy(oracle_trainer)
oracle_trainer_vocalsettech["args"]["train_model"] = oracle_train_model_vocalsettech
oracle_trainer_vocalsettech["args"][
    "train_data_source"
] = train_vocalsettech_data_source
oracle_trainer_vocalsettech["args"]["val_data_source"] = val_vocalsettech_data_source
oracle_trainer_vocalsettech["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsettech


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
] = genre_classification_metrics_vocalsettech


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
] = genre_classification_metrics_vocalsettech


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
] = genre_classification_metrics_vocalsettech


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
] = genre_classification_metrics_vocalsettech


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
] = genre_classification_metrics_vocalsettech


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
] = genre_classification_metrics_vocalsettech


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
] = genre_classification_metrics_vocalsettech


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
] = genre_classification_metrics_vocalsettech

continual_learning_embcenter_trainer_vocalsettech_scenario1 = deepcopy(
    continual_learning_embcenter_trainer
)
continual_learning_embcenter_trainer_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_embcenter_trainer_vocalsettech_scenario1["args"][
    "train_data_source"
] = train_vocalsettech_data_source
continual_learning_embcenter_trainer_vocalsettech_scenario1["args"][
    "val_data_source"
] = val_vocalsettech_data_source
continual_learning_embcenter_trainer_vocalsettech_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsettech


###########               EVALUATORS                ###########

# Scenario 1
oracle_evaluator_vocalsettech = deepcopy(oracle_evaluator)
oracle_evaluator_vocalsettech["args"]["model"] = oracle_train_model_vocalsettech
oracle_evaluator_vocalsettech["args"]["tasks"] = scenario1
oracle_evaluator_vocalsettech["args"]["data_source"] = test_vocalsettech_data_source
oracle_evaluator_vocalsettech["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsettech

continual_learning_evaluator_vocalsettech_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsettech


continual_learning_vq_evaluator_vocalsettech_scenario1 = deepcopy(
    continual_learning_evaluator_vq
)
continual_learning_vq_evaluator_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_vq_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsettech


continual_learning_dkvb_evaluator_vocalsettech_scenario1 = deepcopy(
    continual_learning_evaluator_dkvb
)
continual_learning_dkvb_evaluator_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_dkvb_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsettech


continual_learning_l2p_evaluator_vocalsettech_scenario1 = deepcopy(
    continual_learning_evaluator_l2p
)
continual_learning_l2p_evaluator_vocalsettech_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_l2p_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsettech


continual_learning_embcenter_evaluator_vocalsettech_scenario1 = deepcopy(
    continual_learning_evaluator_embcenter
)
continual_learning_embcenter_evaluator_vocalsettech_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_embcenter_evaluator_vocalsettech_scenario1["args"][
    "data_source"
] = test_vocalsettech_data_source
continual_learning_embcenter_evaluator_vocalsettech_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_vocalsettech


###############################################################
###########               EXPERIMENTS               ###########
###############################################################

###########                BASELINES                ###########

mert95m_base_oracle_vocalsettech_all = {
    "experiment_name": "mert95m_base_oracle_vocalsettech_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": oracle_trainer_vocalsettech,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_vocalsettech,
    },
}


###########            CONTINUAL LEARNING           ###########

# SCENARIO 1

mert95m_finetuning_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_finetuning_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_vocalsettech_scenario1,
    },
}

mert95m_embcenter_cl_vocalsettech_scenario1 = {
    "experiment_name": "mert95m_embcenter_cl_vocalsettech_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "EmbeddingCenter",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_embcenter_trainer_vocalsettech_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_embcenter_evaluator_vocalsettech_scenario1,
    },
}
