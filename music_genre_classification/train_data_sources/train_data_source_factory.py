from music_genre_classification.train_data_sources.train_data_source import (
    TrainDataSource,
)


class TrainDataSourceFactory:
    """
    Factory class for creating train data sources.
    """

    def build(config: dict) -> TrainDataSource:
        if config["name"] == "GtzanDataSource":
            from music_genre_classification.train_data_sources import (
                GtzanDataSource,
            )

            return GtzanDataSource(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
