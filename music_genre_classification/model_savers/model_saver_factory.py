from music_genre_classification.model_savers.model_saver import (
    ModelSaver,
)


class ModelSaverFactory:
    def build(config: dict) -> ModelSaver:
        if config["name"] == "MusicGenderClassificationModelSaver":
            from music_genre_classification.model_savers import (
                MusicGenderClassificationModelSaver,
            )

            return MusicGenderClassificationModelSaver(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
