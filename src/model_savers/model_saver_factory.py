from src.model_savers.model_saver import ModelSaver


class ModelSaverFactory:
    def build(config: dict) -> ModelSaver:
        if config["name"] == "MusicGenreClassificationModelSaver":
            from src.model_savers import (
                MusicGenreClassificationModelSaver,
            )

            return MusicGenreClassificationModelSaver(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
