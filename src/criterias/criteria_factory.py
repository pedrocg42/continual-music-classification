from src.criterias.criteria import Criteria


class CriteriaFactory:
    """
    Factory class for creating criterias.
    """

    def build(config: dict) -> Criteria:
        if config["name"] == "TorchCrossEntropyCriteria":
            from src.criterias import TorchCrossEntropyCriteria

            return TorchCrossEntropyCriteria(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
