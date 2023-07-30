from music_genre_classification.trainers.class_incremental_learning_trainer import (
    ClassIncrementalLearningTrainer,
)
from music_genre_classification.trainers.continual_learning_trainer import (
    ContinualLearningTrainer,
)
from music_genre_classification.trainers.dkvb_continual_learning_trainer import (
    DkvbContinualLearningTrainer,
)
from music_genre_classification.trainers.ewc_continual_learning_trainer import (
    EwcContinualLearningTrainer,
)
from music_genre_classification.trainers.gem_continual_learning_trainer import (
    GemContinualLearningTrainer,
)
from music_genre_classification.trainers.replay_continual_learning_trainer import (
    ReplayContinualLearningTrainer,
)
from music_genre_classification.trainers.trainer import Trainer
from music_genre_classification.trainers.trainer_factory import TrainerFactory
from music_genre_classification.trainers.icarl_continual_learning_trainer import (
    iCaRLContinualLearningTrainer,
)

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
]
