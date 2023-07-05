from music_genre_classification.models.train_model import TrainModel


class TrainModelFactory:
    """
    Factory class for creating train_models.
    """

    def build(config: dict) -> TrainModel:
        if config["name"] == "TorchClassificationModel":
            from music_genre_classification.models import TorchClassificationModel

            return TorchClassificationModel(**config.get("args", {}))
        if config["name"] == "TorchClassIncrementalModel":
            from music_genre_classification.models import TorchClassIncrementalModel

            return TorchClassIncrementalModel(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataModel type: {config['name']}")
