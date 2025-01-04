from torch import nn


class TrainModelFactory:
    """
    Factory class for creating train_models.
    """

    @staticmethod
    def build(config: dict) -> nn.Module:
        if config["name"] == "TorchClassificationModel":
            from src.models import TorchClassificationModel

            return TorchClassificationModel(**config.get("args", {}))
        if config["name"] == "TorchClassIncrementalModel":
            from src.models import TorchClassIncrementalModel

            return TorchClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchMertClassificationModel":
            from src.models import TorchMertClassificationModel

            return TorchMertClassificationModel(**config.get("args", {}))
        if config["name"] == "TorchMertClassIncrementalModel":
            from src.models import TorchMertClassIncrementalModel

            return TorchMertClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchClmrClassificationModel":
            from src.models import TorchClmrClassificationModel

            return TorchClmrClassificationModel(**config.get("args", {}))
        if config["name"] == "TorchClmrClassIncrementalModel":
            from src.models import TorchClmrClassIncrementalModel

            return TorchClmrClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchBottleneckClassificationModel":
            from src.models import (
                TorchBottleneckClassificationModel,
            )

            return TorchBottleneckClassificationModel(**config.get("args", {}))
        if config["name"] == "TorchBottleneckClassIncrementalModel":
            from src.models import (
                TorchBottleneckClassIncrementalModel,
            )

            return TorchBottleneckClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchMertBottleneckClassIncrementalModel":
            from src.models import (
                TorchMertBottleneckClassIncrementalModel,
            )

            return TorchMertBottleneckClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchL2PClassIncrementalModel":
            from src.models import TorchL2PClassIncrementalModel

            return TorchL2PClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchEmbeddingModel":
            from src.models import TorchEmbeddingModel

            return TorchEmbeddingModel(**config.get("args", {}))
        if config["name"] == "TorchEmbeddingCosineModel":
            from src.models import TorchEmbeddingCosineModel

            return TorchEmbeddingCosineModel(**config.get("args", {}))

        raise ValueError(f"Unknown TrainDataModel type: {config['name']}")
