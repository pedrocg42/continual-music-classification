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

###########                TRAINERS                 ###########

# Scenario 1
continual_learning_trainer_scenario1 = deepcopy(continual_learning_trainer)
continual_learning_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_replay_trainer_scenario1 = deepcopy(
    continual_learning_replay_trainer
)
continual_learning_replay_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_vq_trainer_scenario1 = deepcopy(continual_learning_vq_trainer)
continual_learning_vq_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_dkvb_trainer_scenario1 = deepcopy(continual_learning_dkvb_trainer)
continual_learning_dkvb_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_gem_trainer_scenario1 = deepcopy(continual_learning_gem_trainer)
continual_learning_gem_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_ewc_trainer_scenario1 = deepcopy(continual_learning_ewc_trainer)
continual_learning_ewc_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_l2p_trainer_scenario1 = deepcopy(continual_learning_l2p_trainer)
continual_learning_l2p_trainer_scenario1["args"]["tasks"] = scenario1

###########               EVALUATORS                ###########

# Scenario 1
continual_learning_evaluator_scenario1 = deepcopy(evaluator)
continual_learning_evaluator_scenario1["args"]["tasks"] = scenario1

continual_learning_vq_evaluator_scenario1 = deepcopy(continual_learning_evaluator_vq)
continual_learning_vq_evaluator_scenario1["args"]["tasks"] = scenario1

continual_learning_dkvb_evaluator_scenario1 = deepcopy(
    continual_learning_evaluator_dkvb
)
continual_learning_dkvb_evaluator_scenario1["args"]["tasks"] = scenario1


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
        "trainer": oracle_trainer,
    },
    "evaluate": {
        "evaluator": oracle_evaluator,
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
        "trainer": continual_learning_trainer_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_scenario1,
    },
}

mert95m_replay_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_replay_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "Replay",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_replay_trainer_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_scenario1,
    },
}

mert95m_vq_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_vq_cl_gtzan_scenario1",
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

mert95m_dkvb_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_dkvb_cl_gtzan_scenario1",
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

mert95m_gem_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_gem_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "GEM",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_scenario1,
    },
}

mert95m_ewc_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_ewc_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "EWC",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_ewc_trainer_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_scenario1,
    },
}

mert95m_l2p_cl_gtzan_scenario1 = {
    "experiment_name": "mert95m_l2p_cl_gtzan_scenario1",
    "experiment_type": "CL",
    "experiment_subtype": "L2P",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_l2p_trainer_scenario1,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_scenario1,
    },
}
