from music_genre_classification.evaluators import Evaluator


class EvaluatorFactory:
    """
    Factory class for creating trainers.
    """

    @staticmethod
    def build(config: dict) -> Evaluator:
        if config["name"] == "ContinualLearningEvaluator":
            from music_genre_classification.evaluators import ContinualLearningEvaluator

            return ContinualLearningEvaluator(**config["args"])
        else:
            raise Exception("No trainer named %s" % config["name"])
