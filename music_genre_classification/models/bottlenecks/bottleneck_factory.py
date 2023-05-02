from torch.nn import Module


class BottleneckFactory:
    """
    Factory class for creating encoders.
    """

    def build(config: dict) -> Module:
        if config is None:
            return None
        elif config["name"] == "DKVB":
            from music_genre_classification.models.bottlenecks import (
                DKVB,
            )

            return DKVB(**config.get("args", {}))
        elif config["name"] == "VectorQuantizer":
            from music_genre_classification.models.bottlenecks import (
                VectorQuantizer,
            )

            return VectorQuantizer(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown looper type: {config['name']}")
