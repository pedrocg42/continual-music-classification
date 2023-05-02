from music_genre_classification.evaluators import Evaluator


class EvaluatorFactory:
    """
    Factory class for creating trainers.
    """

    @staticmethod
    def build(config: dict) -> Evaluator:
        if config["name"] == "ContinualLearningTasksEvaluatorV2":
            from music_genre_classification.evaluators import (
                ContinualLearningTasksEvaluatorV2,
            )

            return ContinualLearningTasksEvaluatorV2(**config["args"])
        else:
            raise Exception("No trainer named %s" % config["name"])
