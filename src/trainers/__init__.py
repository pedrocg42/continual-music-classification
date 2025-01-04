from src.trainers.class_incremental_learning_trainer import (
    ClassIncrementalLearningTrainer,
)
from src.trainers.continual_learning_trainer import (
    ContinualLearningTrainer,
)
from src.trainers.continual_learning_trainer_embedding_center import (
    ContinualLearningTrainerL2Center,
)
from src.trainers.dkvb_continual_learning_trainer import (
    DkvbContinualLearningTrainer,
)
from src.trainers.ewc_continual_learning_trainer import (
    EwcContinualLearningTrainer,
)
from src.trainers.gem_continual_learning_trainer import (
    GemContinualLearningTrainer,
)
from src.trainers.icarl_continual_learning_trainer import (
    iCaRLContinualLearningTrainer,
)
from src.trainers.replay_continual_learning_trainer import (
    ReplayContinualLearningTrainer,
)
from src.trainers.trainer import Trainer
from src.trainers.trainer_factory import TrainerFactory

__all__ = [
    "TrainerFactory",
    "Trainer",
    "ContinualLearningTrainer",
    "ClassIncrementalLearningTrainer",
    "DkvbContinualLearningTrainer",
    "GemContinualLearningTrainer",
    "EwcContinualLearningTrainer",
    "ReplayContinualLearningTrainer",
    "iCaRLContinualLearningTrainer",
    "ContinualLearningTrainerL2Center",
]
