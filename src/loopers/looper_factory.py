from src.loopers.looper import Looper


class LooperFactory:
    """
    Factory class for creating loopers.
    """

    @staticmethod
    def build(config: dict) -> Looper:
        if config["name"] == "DkvbMusicGenreClassificationLooper":
            from src.loopers import (
                DkvbMusicGenreClassificationLooper,
            )

            return DkvbMusicGenreClassificationLooper(**config.get("args", {}))
        elif config["name"] == "MusicGenreClassificationLooper":
            from src.loopers import (
                MusicGenreClassificationLooper,
            )

            return MusicGenreClassificationLooper(**config.get("args", {}))
        elif config["name"] == "GemMusicGenreClassificationLooper":
            from src.loopers import (
                GemMusicGenreClassificationLooper,
            )

            return GemMusicGenreClassificationLooper(**config.get("args", {}))
        elif config["name"] == "EwcMusicGenreClassificationLooper":
            from src.loopers import (
                EwcMusicGenreClassificationLooper,
            )

            return EwcMusicGenreClassificationLooper(**config.get("args", {}))
        elif config["name"] == "L2PMusicGenreClassificationLooper":
            from src.loopers import (
                L2PMusicGenreClassificationLooper,
            )

            return L2PMusicGenreClassificationLooper(**config.get("args", {}))
        elif config["name"] == "iCaRLMusicGenreClassificationLooper":
            from src.loopers import (
                iCaRLMusicGenreClassificationLooper,
            )

            return iCaRLMusicGenreClassificationLooper(**config.get("args", {}))
        elif config["name"] == "MusicContinualLearningEmbeddingLooper":
            from src.loopers import (
                MusicContinualLearningEmbeddingLooper,
            )

            return MusicContinualLearningEmbeddingLooper(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown looper type: {config['name']}")
