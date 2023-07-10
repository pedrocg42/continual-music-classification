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
from music_genre_classification.trainers.trainer import Trainer
from music_genre_classification.trainers.trainer_factory import TrainerFactory

__all__ = [
    "TrainerFactory",
    "Trainer",
    "ContinualLearningTrainer",
    "ClassIncrementalLearningTrainer",
    "DkvbContinualLearningTrainer",
    "GemContinualLearningTrainer",
    "EwcContinualLearningTrainer",
]
