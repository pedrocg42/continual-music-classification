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

# Baselines
continual_learning_trainer_all = deepcopy(continual_learning_trainer)
continual_learning_trainer_all["args"]["tasks"] = all_tasks

continual_learning_vq_trainer_all = deepcopy(continual_learning_vq_trainer)
continual_learning_vq_trainer_all["args"]["tasks"] = all_tasks


continual_learning_dkvb_trainer_all = deepcopy(continual_learning_dkvb_trainer)
continual_learning_dkvb_trainer_all["args"]["tasks"] = all_tasks

continual_learning_gem_trainer_all = deepcopy(continual_learning_gem_trainer)
continual_learning_gem_trainer_all["args"]["tasks"] = all_tasks

# Scenario 1
continual_learning_trainer_scenario1 = deepcopy(continual_learning_trainer)
continual_learning_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_vq_trainer_scenario1 = deepcopy(continual_learning_vq_trainer_all)
continual_learning_vq_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_dkvb_trainer_scenario1 = deepcopy(
    continual_learning_dkvb_trainer_all
)
continual_learning_dkvb_trainer_scenario1["args"]["tasks"] = scenario1

continual_learning_gem_trainer_scenario1 = deepcopy(continual_learning_gem_trainer)
continual_learning_gem_trainer_scenario1["args"]["tasks"] = scenario1

###########               EVALUATORS                ###########

# Baselines
continual_learning_evaluator_all = deepcopy(continual_learning_evaluator)
continual_learning_evaluator_all["args"]["tasks"] = all_tasks

continual_learning_vq_evaluator_all = deepcopy(continual_learning_evaluator_vq)
continual_learning_vq_evaluator_all["args"]["tasks"] = all_tasks

continual_learning_dkvb_evaluator_all = deepcopy(continual_learning_evaluator_dkvb)
continual_learning_dkvb_evaluator_all["args"]["tasks"] = all_tasks

# Scenario 1
continual_learning_evaluator_scenario1 = deepcopy(continual_learning_evaluator)
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
        "trainer": continual_learning_trainer_all,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_all,
    },
}

mert95m_vq_oracle_gtzan_all = {
    "experiment_name": "mert95m_vq_oracle_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "oracle",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_vq_trainer_all,
    },
    "evaluate": {
        "evaluator": continual_learning_vq_evaluator_all,
    },
}

mert95m_dkvb_oracle_gtzan_all = {
    "experiment_name": "mert95m_dkvb_oracle_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_dkvb_trainer_all,
    },
    "evaluate": {
        "evaluator": continual_learning_dkvb_evaluator_all,
    },
}

mert95m_gem_oracle_gtzan_all = {
    "experiment_name": "mert95m_gem_oracle_gtzan_all",
    "experiment_type": "Baseline",
    "experiment_subtype": "Oracle",
    "num_cross_val_splits": num_cross_val_splits,
    # data
    "train": {
        "trainer": continual_learning_gem_trainer_all,
    },
    "evaluate": {
        "evaluator": continual_learning_evaluator_all,
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
