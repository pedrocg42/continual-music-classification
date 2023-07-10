from music_genre_classification.optimizers.optimizer import Optimizer


class OptimizerFactory:
    """
    Factory class for creating optimizers.
    """

    def build(config: dict) -> Optimizer:
        if config["name"] == "TorchSgdOptimizer":
            from music_genre_classification.optimizers import TorchSgdOptimizer

            return TorchSgdOptimizer(**config.get("args", {}))
        elif config["name"] == "TorchAdamWOptimizer":
            from music_genre_classification.optimizers import TorchAdamWOptimizer

            return TorchAdamWOptimizer(**config.get("args", {}))
        elif config["name"] == "GemOptimizer":
            from music_genre_classification.optimizers import GemOptimizer

            return GemOptimizer(**config.get("args", {}))
        elif config["name"] == "EwcOptimizer":
            from music_genre_classification.optimizers import EwcOptimizer

            return EwcOptimizer(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown Optimizer type: {config['name']}")
