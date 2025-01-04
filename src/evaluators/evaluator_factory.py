from src.evaluators import Evaluator


class EvaluatorFactory:
    """
    Factory class for creating trainers.
    """

    @staticmethod
    def build(config: dict) -> Evaluator:
        if config["name"] == "ContinualLearningEvaluator":
            from src.evaluators import ContinualLearningEvaluator

            return ContinualLearningEvaluator(**config["args"])
        if config["name"] == "ClassIncrementalLearningEvaluator":
            from src.evaluators import (
                ClassIncrementalLearningEvaluator,
            )

            return ClassIncrementalLearningEvaluator(**config["args"])
        if config["name"] == "ClassIncrementalLearningOracleEvaluator":
            from src.evaluators import (
                ClassIncrementalLearningOracleEvaluator,
            )

            return ClassIncrementalLearningOracleEvaluator(**config["args"])
        if config["name"] == "ClassIncrementalLearningDKVBEvaluator":
            from src.evaluators import (
                ClassIncrementalLearningDKVBEvaluator,
            )

            return ClassIncrementalLearningDKVBEvaluator(**config["args"])
        if config["name"] == "ClassIncrementalLearningL2PEvaluator":
            from src.evaluators import (
                ClassIncrementalLearningL2PEvaluator,
            )

            return ClassIncrementalLearningL2PEvaluator(**config["args"])
        if config["name"] == "ClassIncrementalLearningL2CenterEvaluator":
            from src.evaluators import (
                ClassIncrementalLearningL2CenterEvaluator,
            )

            return ClassIncrementalLearningL2CenterEvaluator(**config["args"])

        raise Exception("No evaluator named {}".format(config["name"]))
