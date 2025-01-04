from src.optimizers.optimizer import Optimizer


class OptimizerFactory:
    """
    Factory class for creating optimizers.
    """

    def build(config: dict) -> Optimizer:
        if config["name"] == "TorchSgdOptimizer":
            from src.optimizers import TorchSgdOptimizer

            return TorchSgdOptimizer(**config.get("args", {}))
        elif config["name"] == "TorchAdamWOptimizer":
            from src.optimizers import TorchAdamWOptimizer

            return TorchAdamWOptimizer(**config.get("args", {}))
        elif config["name"] == "TorchAdamOptimizer":
            from src.optimizers import TorchAdamOptimizer

            return TorchAdamOptimizer(**config.get("args", {}))
        elif config["name"] == "GemOptimizer":
            from src.optimizers import GemOptimizer

            return GemOptimizer(**config.get("args", {}))
        elif config["name"] == "EwcOptimizer":
            from src.optimizers import EwcOptimizer

            return EwcOptimizer(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown Optimizer type: {config['name']}")
