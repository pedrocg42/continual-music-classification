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
        if config["name"] == "ClassIncrementalLearningEvaluator":
            from music_genre_classification.evaluators import (
                ClassIncrementalLearningEvaluator,
            )

            return ClassIncrementalLearningEvaluator(**config["args"])
        if config["name"] == "ClassIncrementalLearningOracleEvaluator":
            from music_genre_classification.evaluators import (
                ClassIncrementalLearningOracleEvaluator,
            )

            return ClassIncrementalLearningOracleEvaluator(**config["args"])
        if config["name"] == "ClassIncrementalLearningDKVBEvaluator":
            from music_genre_classification.evaluators import (
                ClassIncrementalLearningDKVBEvaluator,
            )

            return ClassIncrementalLearningDKVBEvaluator(**config["args"])
        if config["name"] == "ClassIncrementalLearningL2PEvaluator":
            from music_genre_classification.evaluators import (
                ClassIncrementalLearningL2PEvaluator,
            )

            return ClassIncrementalLearningL2PEvaluator(**config["args"])
        if config["name"] == "ClassIncrementalLearningEmbeddingCenterEvaluator":
            from music_genre_classification.evaluators import (
                ClassIncrementalLearningEmbeddingCenterEvaluator,
            )

            return ClassIncrementalLearningEmbeddingCenterEvaluator(**config["args"])

        raise Exception("No evaluator named %s" % config["name"])
