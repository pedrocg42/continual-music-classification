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
        if config["name"] == "DkvbContinualLearningTrainer":
            from music_genre_classification.trainers import DkvbContinualLearningTrainer

            return DkvbContinualLearningTrainer(**config["args"])
        if config["name"] == "GemContinualLearningTrainer":
            from music_genre_classification.trainers import GemContinualLearningTrainer

            return GemContinualLearningTrainer(**config["args"])
        else:
            raise Exception("No trainer named %s" % config["name"])
