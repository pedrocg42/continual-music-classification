from music_genre_classification.loopers.looper import Looper


class LooperFactory:
    """
    Factory class for creating loopers.
    """

    @staticmethod
    def build(config: dict) -> Looper:
        if config["name"] == "DkvbMusicGenreClassificationLooper":
            from music_genre_classification.loopers import (
                DkvbMusicGenreClassificationLooper,
            )

            return DkvbMusicGenreClassificationLooper(**config.get("args", {}))
        elif config["name"] == "MusicGenreClassificationLooper":
            from music_genre_classification.loopers import (
                MusicGenreClassificationLooper,
            )

            return MusicGenreClassificationLooper(**config.get("args", {}))
        elif config["name"] == "GemMusicGenreClassificationLooper":
            from music_genre_classification.loopers import (
                GemMusicGenreClassificationLooper,
            )

            return GemMusicGenreClassificationLooper(**config.get("args", {}))
        elif config["name"] == "EwcMusicGenreClassificationLooper":
            from music_genre_classification.loopers import (
                EwcMusicGenreClassificationLooper,
            )

            return EwcMusicGenreClassificationLooper(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown looper type: {config['name']}")
