from music_genre_classification.trainers import Trainer
class TrainerFactory:
    """
    Factory class for creating trainers.
    """

    @staticmethod
    def build(trainer_name, args) -> Trainer:
        """
        Create trainer.
        :param trainer_name: Name of the trainer.
        :param args: Arguments for the trainer.
        """
        if trainer_name == "ContinualLearningTrainer":
            from music_genre_classification.trainers import ContinualLearningTrainer

            return ContinualLearningTrainer(**args)
        else:
            raise Exception("No trainer named %s" % trainer_name)
