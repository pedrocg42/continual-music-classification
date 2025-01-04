from src.train_data_sources.train_data_source import (
    TrainDataSource,
)


class TrainDataSourceFactory:
    """
    Factory class for creating train data sources.
    """

    @staticmethod
    def build(config: dict) -> TrainDataSource:
        if config["name"] == "GtzanDataSource":
            from src.train_data_sources import GtzanDataSource

            return GtzanDataSource(**config.get("args", {}))
        if config["name"] == "VocalSetSingerDataSource":
            from src.train_data_sources import (
                VocalSetSingerDataSource,
            )

            return VocalSetSingerDataSource(**config.get("args", {}))
        if config["name"] == "VocalSetTechDataSource":
            from src.train_data_sources import (
                VocalSetTechDataSource,
            )

            return VocalSetTechDataSource(**config.get("args", {}))
        if config["name"] == "NSynthInstrumentTechDataSource":
            from src.train_data_sources import (
                NSynthInstrumentTechDataSource,
            )

            return NSynthInstrumentTechDataSource(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
