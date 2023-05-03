from experiments.components import *
from copy import deepcopy

###############################################################
###########                SCENARIOS                ###########
###############################################################

all_tasks = ["all"]

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

# Trainers
continual_learning_trainer_all = deepcopy(continual_learning_trainer)
continual_learning_trainer_all["args"]["tasks"] = all_tasks

continual_learning_vq_trainer_all = deepcopy(continual_learning_trainer)
continual_learning_vq_trainer_all["args"]["tasks"] = all_tasks
continual_learning_vq_trainer_all["name"] = "DkvbContinualLearningTrainer"
continual_learning_vq_trainer_all["args"]["epochs_keys_init"] = epochs_keys_init
continual_learning_vq_trainer_all["args"]["freeze_decoder_after_first_epoch"] = False
continual_learning_vq_trainer_all["args"]["looper"][
    "name"
] = "DkvbMusicGenreClassificationLooper"
continual_learning_vq_trainer_all["args"]["looper"]["args"][
    "train_model"
] = train_model_vq

continual_learning_dkvb_trainer_all = deepcopy(continual_learning_trainer)
continual_learning_dkvb_trainer_all["args"]["tasks"] = all_tasks
continual_learning_dkvb_trainer_all["name"] = "DkvbContinualLearningTrainer"
continual_learning_dkvb_trainer_all["args"]["epochs_keys_init"] = epochs_keys_init
continual_learning_dkvb_trainer_all["args"]["freeze_decoder_after_first_epoch"] = False
continual_learning_dkvb_trainer_all["args"]["looper"][
    "name"
] = "DkvbMusicGenreClassificationLooper"
continual_learning_dkvb_trainer_all["args"]["looper"]["args"][
    "train_model"
] = train_model_dkvb

continual_learning_trainer_scenario1 = deepcopy(continual_learning_trainer)
continual_learning_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_vq_trainer_scenario1 = deepcopy(continual_learning_vq_trainer_all)
continual_learning_vq_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_dkvb_trainer_scenario1 = deepcopy(
    continual_learning_dkvb_trainer_all
)
continual_learning_dkvb_trainer_scenario1["args"]["tasks"] = scenario1

# Evaluators
continual_learning_evaluator_all = deepcopy(continual_learning_evaluator)
continual_learning_evaluator_all["args"]["train_tasks"] = all_tasks
continual_learning_evaluator_all["args"]["test_tasks"] = all_tasks

continual_learning_vq_evaluator_all = deepcopy(continual_learning_evaluator)
continual_learning_vq_evaluator_all["args"]["model"] = train_model_vq

continual_learning_dkvb_evaluator_all = deepcopy(continual_learning_evaluator)
continual_learning_dkvb_evaluator_all["args"]["model"] = train_model_dkvb

continual_learning_evaluator_scenario1 = deepcopy(continual_learning_evaluator)
continual_learning_evaluator_scenario1["args"]["train_tasks"] = scenario1
continual_learning_evaluator_scenario1["args"]["test_tasks"] = scenario1

continual_learning_vq_evaluator_scenario1 = deepcopy(
    continual_learning_evaluator_scenario1
)
continual_learning_vq_evaluator_scenario1["args"]["model"] = train_model_vq

continual_learning_dkvb_evaluator_scenario1 = deepcopy(
    continual_learning_evaluator_scenario1
)
continual_learning_dkvb_evaluator_scenario1["args"]["model"] = train_model_dkvb

###############################################################
###########               EXPERIMENTS               ###########
###############################################################

# Baselines joint

Mert95m_joint_gtzan_all = {
    "experiment_name": "Mert95m_joint_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_trainer_all,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_all,
    },
}

Mert95mVq_joint_gtzan_all = {
    "experiment_name": "Mert95mVq_joint_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_all,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_all,
    },
}

Mert95mDkvb_joint_gtzan_all = {
    "experiment_name": "Mert95mDkvb_joint_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Joint",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_all,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_all,
    },
}

# Continual Learning scenario 1
Mert95m_cl_gtzan_scenario1 = {
    "experiment_name": "Mert95m_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Baseline",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_trainer_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_scenario1,
    },
}

Mert95mVq_cl_gtzan_scenario1 = {
    "experiment_name": "Mert95mVq_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "VQ",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_scenario1,
    },
}

Mert95mDkvb_cl_gtzan_scenario1 = {
    "experiment_name": "Mert95mDkvb_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "DKVB",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_scenario1,
    },
}
