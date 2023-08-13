from copy import deepcopy

from experiments.components import *

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

# Data sources
train_nsynthinstrument_data_source = {
    "name": "NSynthInstrumentTechDataSource",
    "args": {
        "split": "train",
        "chunk_length": 3,
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
genre_classification_metrics_nsynthinstrument = [
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

# Scenario 1
oracle_trainer_nsynthinstrument = deepcopy(oracle_trainer)
oracle_trainer_nsynthinstrument["args"][
    "train_model"
] = oracle_train_model_nsynthinstrument
oracle_trainer_nsynthinstrument["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
oracle_trainer_nsynthinstrument["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
oracle_trainer_nsynthinstrument["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
oracle_trainer_nsynthinstrument["args"]["early_stopping_patience"] = 5


continual_learning_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_trainer
)
continual_learning_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_trainer_nsynthinstrument_scenario1["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
continual_learning_trainer_nsynthinstrument_scenario1["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
continual_learning_trainer_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
continual_learning_trainer_nsynthinstrument_scenario1["args"][
    "early_stopping_patience"
] = 5


continual_learning_replay_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_replay_trainer
)
continual_learning_replay_trainer_nsynthinstrument_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_replay_trainer_nsynthinstrument_scenario1["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
continual_learning_replay_trainer_nsynthinstrument_scenario1["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
continual_learning_replay_trainer_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
continual_learning_replay_trainer_nsynthinstrument_scenario1["args"][
    "early_stopping_patience"
] = 5


continual_learning_icarl_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_icarl_trainer
)
continual_learning_icarl_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_icarl_trainer_nsynthinstrument_scenario1["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
continual_learning_icarl_trainer_nsynthinstrument_scenario1["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
continual_learning_icarl_trainer_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
continual_learning_icarl_trainer_nsynthinstrument_scenario1["args"][
    "early_stopping_patience"
] = 5


continual_learning_vq_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_vq_trainer
)
continual_learning_vq_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_trainer_nsynthinstrument_scenario1["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
continual_learning_vq_trainer_nsynthinstrument_scenario1["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
continual_learning_vq_trainer_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
continual_learning_vq_trainer_nsynthinstrument_scenario1["args"][
    "early_stopping_patience"
] = 5


continual_learning_dkvb_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_dkvb_trainer
)
continual_learning_dkvb_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_dkvb_trainer_nsynthinstrument_scenario1["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
continual_learning_dkvb_trainer_nsynthinstrument_scenario1["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
continual_learning_dkvb_trainer_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
continual_learning_dkvb_trainer_nsynthinstrument_scenario1["args"][
    "early_stopping_patience"
] = 5


continual_learning_gem_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_gem_trainer
)
continual_learning_gem_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_gem_trainer_nsynthinstrument_scenario1["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
continual_learning_gem_trainer_nsynthinstrument_scenario1["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
continual_learning_gem_trainer_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
continual_learning_gem_trainer_nsynthinstrument_scenario1["args"][
    "early_stopping_patience"
] = 5


continual_learning_ewc_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_ewc_trainer
)
continual_learning_ewc_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_ewc_trainer_nsynthinstrument_scenario1["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
continual_learning_ewc_trainer_nsynthinstrument_scenario1["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
continual_learning_ewc_trainer_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
continual_learning_ewc_trainer_nsynthinstrument_scenario1["args"][
    "early_stopping_patience"
] = 5


continual_learning_l2p_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_l2p_trainer
)
continual_learning_l2p_trainer_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_trainer_nsynthinstrument_scenario1["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
continual_learning_l2p_trainer_nsynthinstrument_scenario1["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
continual_learning_l2p_trainer_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
continual_learning_l2p_trainer_nsynthinstrument_scenario1["args"][
    "early_stopping_patience"
] = 5

continual_learning_embcenter_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_embcenter_trainer
)
continual_learning_embcenter_trainer_nsynthinstrument_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_embcenter_trainer_nsynthinstrument_scenario1["args"][
    "train_data_source"
] = train_nsynthinstrument_data_source
continual_learning_embcenter_trainer_nsynthinstrument_scenario1["args"][
    "val_data_source"
] = val_nsynthinstrument_data_source
continual_learning_embcenter_trainer_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument
continual_learning_embcenter_trainer_nsynthinstrument_scenario1["args"][
    "early_stopping_patience"
] = 5

continual_learning_embcentercosine_trainer_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_embcenter_trainer_nsynthinstrument_scenario1
)
continual_learning_embcentercosine_trainer_nsynthinstrument_scenario1["args"][
    "train_model"
] = train_model_embcentercosine

###########               EVALUATORS                ###########

# Scenario 1
oracle_evaluator_nsynthinstrument = deepcopy(oracle_evaluator)
oracle_evaluator_nsynthinstrument["args"]["model"] = oracle_train_model_nsynthinstrument
oracle_evaluator_nsynthinstrument["args"]["tasks"] = scenario1
oracle_evaluator_nsynthinstrument["args"][
    "data_source"
] = test_nsynthinstrument_data_source
oracle_evaluator_nsynthinstrument["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument

continual_learning_evaluator_nsynthinstrument_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_evaluator_nsynthinstrument_scenario1["args"][
    "data_source"
] = test_nsynthinstrument_data_source
continual_learning_evaluator_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument


continual_learning_vq_evaluator_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_evaluator_vq
)
continual_learning_vq_evaluator_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_vq_evaluator_nsynthinstrument_scenario1["args"][
    "data_source"
] = test_nsynthinstrument_data_source
continual_learning_vq_evaluator_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument


continual_learning_dkvb_evaluator_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_evaluator_dkvb
)
continual_learning_dkvb_evaluator_nsynthinstrument_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_dkvb_evaluator_nsynthinstrument_scenario1["args"][
    "data_source"
] = test_nsynthinstrument_data_source
continual_learning_dkvb_evaluator_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument


continual_learning_l2p_evaluator_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_evaluator_l2p
)
continual_learning_l2p_evaluator_nsynthinstrument_scenario1["args"]["tasks"] = scenario1
continual_learning_l2p_evaluator_nsynthinstrument_scenario1["args"][
    "data_source"
] = test_nsynthinstrument_data_source
continual_learning_l2p_evaluator_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument


continual_learning_embcenter_evaluator_nsynthinstrument_scenario1 = deepcopy(
    continual_learning_evaluator_embcenter
)
continual_learning_embcenter_evaluator_nsynthinstrument_scenario1["args"][
    "tasks"
] = scenario1
continual_learning_embcenter_evaluator_nsynthinstrument_scenario1["args"][
    "data_source"
] = test_nsynthinstrument_data_source
continual_learning_embcenter_evaluator_nsynthinstrument_scenario1["args"][
    "metrics_config"
] = genre_classification_metrics_nsynthinstrument


###############################################################
###########               EXPERIMENTS               ###########
###############################################################

###########                BASELINES                ###########

mert95m_base_oracle_nsynthinstrument_all = {
    "experiment_name": "mert95m_base_oracle_nsynthinstrument_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    # data
    "train": {
        "trainer": oracle_trainer_nsynthinstrument,
    },
    "evaluate": {
        "evaluator": oracle_evaluator_nsynthinstrument,
    },
}


###########            CONTINUAL LEARNING           ###########

# SCENARIO 1

mert95m_finetuning_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_finetuning_cl_nsynthinstrument_scenario1",
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

mert95m_replay_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_replay_cl_nsynthinstrument_scenario1",
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

mert95m_icarl_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_icarl_cl_nsynthinstrument_scenario1",
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

mert95m_vq_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_vq_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_nsynthinstrument_scenario1,
    },
}

mert95m_dkvb_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_dkvb_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_nsynthinstrument_scenario1,
    },
}

mert95m_gem_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_gem_cl_nsynthinstrument_scenario1",
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

mert95m_ewc_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_ewc_cl_nsynthinstrument_scenario1",
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

mert95m_l2p_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_l2p_cl_nsynthinstrument_scenario1",
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

mert95m_embcenter_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_embcenter_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "EmbeddingCenter",
    # data
    "train": {
        "trainer": continual_learning_embcenter_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_embcenter_evaluator_nsynthinstrument_scenario1,
    },
}

mert95m_embcentercosine_cl_nsynthinstrument_scenario1 = {
    "experiment_name": "mert95m_embcentercosine_cl_nsynthinstrument_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "CenterCosine",
    # data
    "train": {
        "trainer": continual_learning_embcentercosine_trainer_nsynthinstrument_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_embcenter_evaluator_nsynthinstrument_scenario1,
    },
}
