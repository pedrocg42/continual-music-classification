from music_genre_classification.optimizers.optimizer import Optimizer


class OptimizerFactory:
    """
    Factory class for creating optimizers.
    """

    def build(config: dict) -> Optimizer:
        if config["name"] == "TorchAdamWOptimizer":
            from music_genre_classification.optimizers import (
                TorchAdamWOptimizer,
            )

            return TorchAdamWOptimizer(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
