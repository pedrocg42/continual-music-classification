from music_genre_classification.trainers import Trainer


class TrainerFactory:
    """
    Factory class for creating trainers.
    """

    @staticmethod
    def build(config: dict) -> Trainer:
        """
        Create trainer.
        :param trainer_name: Name of the trainer.
        :param args: Arguments for the trainer.
        """
        if config["name"] == "ContinualLearningTrainer":
            from music_genre_classification.trainers import ContinualLearningTrainer

            return ContinualLearningTrainer(**config["args"])
        if config["name"] == "ClassIncrementalLearningTrainer":
            from music_genre_classification.trainers import (
                ClassIncrementalLearningTrainer,
            )

            return ClassIncrementalLearningTrainer(**config["args"])
        if config["name"] == "DkvbContinualLearningTrainer":
            from music_genre_classification.trainers import DkvbContinualLearningTrainer

            return DkvbContinualLearningTrainer(**config["args"])
        if config["name"] == "GemContinualLearningTrainer":
            from music_genre_classification.trainers import GemContinualLearningTrainer

            return GemContinualLearningTrainer(**config["args"])
        if config["name"] == "EwcContinualLearningTrainer":
            from music_genre_classification.trainers import EwcContinualLearningTrainer

            return EwcContinualLearningTrainer(**config["args"])
        if config["name"] == "ReplayContinualLearningTrainer":
            from music_genre_classification.trainers import (
                ReplayContinualLearningTrainer,
            )

            return ReplayContinualLearningTrainer(**config["args"])
        if config["name"] == "iCaRLContinualLearningTrainer":
            from music_genre_classification.trainers import (
                iCaRLContinualLearningTrainer,
            )

            return iCaRLContinualLearningTrainer(**config["args"])
        if config["name"] == "ContinualLearningTrainerL2Center":
            from music_genre_classification.trainers import (
                ContinualLearningTrainerL2Center,
            )

            return ContinualLearningTrainerL2Center(**config["args"])

        raise Exception("No trainer named %s" % config["name"])
