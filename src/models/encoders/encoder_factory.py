from torch.nn import Module


class EncoderFactory:
    """
    Factory class for creating encoders.
    """

    def build(config: dict) -> Module:
        if config["name"] == "MertEncoder":
            from src.models.encoders import MertEncoder

            return MertEncoder(**config.get("args", {}))
        if config["name"] == "MertEncoderL2P":
            from src.models.encoders import MertEncoderL2P

            return MertEncoderL2P(**config.get("args", {}))
        if config["name"] == "ClmrEncoder":
            from src.models.encoders import ClmrEncoder

            return ClmrEncoder(**config.get("args", {}))
        elif config["name"] == "ResNet50DinoEncoder":
            from src.models.encoders import ResNet50DinoEncoder

            return ResNet50DinoEncoder(**config.get("args", {}))
        elif config["name"] == "ResNet50Encoder":
            from src.models.encoders import ResNet50Encoder

            return ResNet50Encoder(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown encoder type: {config['name']}")
