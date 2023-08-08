from torch import nn


class TrainModelFactory:
    """
    Factory class for creating train_models.
    """

    @staticmethod
    def build(config: dict) -> nn.Module:
        if config["name"] == "TorchClassificationModel":
            from music_genre_classification.models import TorchClassificationModel

            return TorchClassificationModel(**config.get("args", {}))
        if config["name"] == "TorchClassIncrementalModel":
            from music_genre_classification.models import TorchClassIncrementalModel

            return TorchClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchMertClassificationModel":
            from music_genre_classification.models import TorchMertClassificationModel

            return TorchMertClassificationModel(**config.get("args", {}))
        if config["name"] == "TorchMertClassIncrementalModel":
            from music_genre_classification.models import TorchMertClassIncrementalModel

            return TorchMertClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchBottleneckClassificationModel":
            from music_genre_classification.models import (
                TorchBottleneckClassificationModel,
            )

            return TorchBottleneckClassificationModel(**config.get("args", {}))
        if config["name"] == "TorchBottleneckClassIncrementalModel":
            from music_genre_classification.models import (
                TorchBottleneckClassIncrementalModel,
            )

            return TorchBottleneckClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchMertBottleneckClassIncrementalModel":
            from music_genre_classification.models import (
                TorchMertBottleneckClassIncrementalModel,
            )

            return TorchMertBottleneckClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchL2PClassIncrementalModel":
            from music_genre_classification.models import TorchL2PClassIncrementalModel

            return TorchL2PClassIncrementalModel(**config.get("args", {}))
        if config["name"] == "TorchEmbeddingModel":
            from music_genre_classification.models import TorchEmbeddingModel

            return TorchEmbeddingModel(**config.get("args", {}))

        raise ValueError(f"Unknown TrainDataModel type: {config['name']}")
