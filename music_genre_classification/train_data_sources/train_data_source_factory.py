from music_genre_classification.train_data_sources.train_data_source import (
    TrainDataSource,
)


class TrainDataSourceFactory:
    """
    Factory class for creating train data sources.
    """

    def build(config: dict) -> TrainDataSource:
        if config["name"] == "GtzanDataSource":
            from music_genre_classification.train_data_sources import GtzanDataSource

            return GtzanDataSource(**config.get("args", {}))
        if config["name"] == "VocalSetSingerDataSource":
            from music_genre_classification.train_data_sources import (
                VocalSetSingerDataSource,
            )

            return VocalSetSingerDataSource(**config.get("args", {}))
        if config["name"] == "VocalSetTechDataSource":
            from music_genre_classification.train_data_sources import (
                VocalSetTechDataSource,
            )

            return VocalSetTechDataSource(**config.get("args", {}))
        if config["name"] == "NSynthInstrumentTechDataSource":
            from music_genre_classification.train_data_sources import (
                NSynthInstrumentTechDataSource,
            )

            return NSynthInstrumentTechDataSource(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
