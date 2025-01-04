from src.trainers import Trainer


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
            from src.trainers import ContinualLearningTrainer

            return ContinualLearningTrainer(**config["args"])
        if config["name"] == "ClassIncrementalLearningTrainer":
            from src.trainers import (
                ClassIncrementalLearningTrainer,
            )

            return ClassIncrementalLearningTrainer(**config["args"])
        if config["name"] == "DkvbContinualLearningTrainer":
            from src.trainers import DkvbContinualLearningTrainer

            return DkvbContinualLearningTrainer(**config["args"])
        if config["name"] == "GemContinualLearningTrainer":
            from src.trainers import GemContinualLearningTrainer

            return GemContinualLearningTrainer(**config["args"])
        if config["name"] == "EwcContinualLearningTrainer":
            from src.trainers import EwcContinualLearningTrainer

            return EwcContinualLearningTrainer(**config["args"])
        if config["name"] == "ReplayContinualLearningTrainer":
            from src.trainers import (
                ReplayContinualLearningTrainer,
            )

            return ReplayContinualLearningTrainer(**config["args"])
        if config["name"] == "iCaRLContinualLearningTrainer":
            from src.trainers import (
                iCaRLContinualLearningTrainer,
            )

            return iCaRLContinualLearningTrainer(**config["args"])
        if config["name"] == "ContinualLearningTrainerL2Center":
            from src.trainers import (
                ContinualLearningTrainerL2Center,
            )

            return ContinualLearningTrainerL2Center(**config["args"])

        raise Exception("No trainer named {}".format(config["name"]))
