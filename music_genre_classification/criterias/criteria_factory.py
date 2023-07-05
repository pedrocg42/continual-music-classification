from music_genre_classification.criterias.criteria import Criteria


class CriteriaFactory:
    """
    Factory class for creating criterias.
    """

    def build(config: dict) -> Criteria:
        if config["name"] == "TorchCrossEntropyCriteria":
            from music_genre_classification.criterias import TorchCrossEntropyCriteria

            return TorchCrossEntropyCriteria(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
