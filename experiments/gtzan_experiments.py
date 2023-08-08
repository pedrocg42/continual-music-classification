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
    ["blues", "classical"],
    ["country", "disco"],
    ["hiphop", "jazz"],
    ["metal", "pop"],
    ["reggae", "rock"],
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
        "chunk_length": 3,
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

oracle_train_model_gtzan = {
    "name": "TorchMertClassificationModel",
    "args": {"num_classes": num_classes},
}

# Metrics
genre_classification_metrics_gtzan = [
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

# Scenario 1
oracle_trainer_gtzan = deepcopy(oracle_trainer)
oracle_trainer_gtzan["args"]["train_model"] = oracle_train_model_gtzan
oracle_trainer_gtzan["args"]["train_data_source"] = train_gtzan_data_source
oracle_trainer_gtzan["args"]["val_data_source"] = train_gtzan_data_source
oracle_trainer_gtzan["args"]["metrics_config"] = oracle_trainer_gtzan


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
] = genre_classification_metrics_gtzan


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
] = genre_classification_metrics_gtzan


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
] = genre_classification_metrics_gtzan


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
] = genre_classification_metrics_gtzan


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
] = genre_classification_metrics_gtzan


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
] = genre_classification_metrics_gtzan


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
] = genre_classification_metrics_gtzan


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
] = genre_classification_metrics_gtzan


continual_learning_embcenter_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_embcenter_trainer
)
continual_learning_embcenter_trainer_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_embcenter_trainer_gtzan_scenario1["args"][
    "train_data_source"
] = train_gtzan_data_source
continual_learning_embcenter_trainer_gtzan_scenario1["args"][
    "val_data_source"
] = val_gtzan_data_source
continual_learning_embcenter_trainer_gtzan_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_gtzan

continual_learning_embcentercosine_trainer_gtzan_scenario1 = deepcopy(
    continual_learning_embcenter_trainer_gtzan_scenario1
)
continual_learning_embcentercosine_trainer_gtzan_scenario1["args"][
    "train_model"
] = train_model_embcentercosine

###########               EVALUATORS                ###########

# Scenario 1
oracle_evaluator_gtzan = deepcopy(oracle_evaluator)
oracle_evaluator_gtzan["args"]["model"] = oracle_train_model_gtzan
oracle_evaluator_gtzan["args"]["tasks"] = scenario1
oracle_evaluator_gtzan["args"]["data_source"] = test_gtzan_data_source
oracle_evaluator_gtzan["args"]["metrics_config"] = genre_classification_metrics_gtzan

continual_learning_evaluator_gtzan_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_gtzan


continual_learning_vq_evaluator_gtzan_scenario1 = deepcopy(
    continual_learning_evaluator_vq
)
continual_learning_vq_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_vq_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_gtzan


continual_learning_dkvb_evaluator_gtzan_scenario1 = deepcopy(
    continual_learning_evaluator_dkvb
)
continual_learning_dkvb_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_dkvb_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_gtzan


continual_learning_l2p_evaluator_gtzan_scenario1 = deepcopy(
    continual_learning_evaluator_l2p
)
continual_learning_l2p_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_l2p_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_gtzan


continual_learning_embcenter_evaluator_gtzan_scenario1 = deepcopy(
    continual_learning_evaluator_embcenter
)
continual_learning_embcenter_evaluator_gtzan_scenario1["args"]["tasks"] = scenario1
continual_learning_embcenter_evaluator_gtzan_scenario1["args"][
    "data_source"
] = test_gtzan_data_source
continual_learning_embcenter_evaluator_gtzan_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_gtzan


###############################################################
###########               EXPERIMENTS               ###########
###############################################################

###########                BASELINES                ###########

mert95m_base_oracle_gtzan_all = {
    "experiment_name": "mert95m_base_oracle_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": oracle_trainer_gtzan,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_gtzan,
    },
}


###########            CONTINUAL LEARNING           ###########

# SCENARIO 1

mert95m_finetuning_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_finetuning_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Finetuning",
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
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
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_l2p_evaluator_gtzan_scenario1,
    },
}

mert95m_embcenter_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_embcenter_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "EmbeddingCenter",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_embcenter_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_embcenter_evaluator_gtzan_scenario1,
    },
}

mert95m_embcentercosine_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_embcentercosine_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "CenterCosine",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_embcentercosine_trainer_gtzan_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_embcenter_evaluator_gtzan_scenario1,
    },
}
