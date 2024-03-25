from music_genre_classification.train_data_transforms.train_data_transform import (
    TrainDataTransform,
)


class TrainDataTransformFactory:
    def build(config: dict) -> TrainDataTransform:
        if config["name"] == "MertDataTransform":
            from music_genre_classification.train_data_transforms import (
                MertDataTransform,
            )

            return MertDataTransform(**config.get("args", {}))
        if config["name"] == "ResamplerDataTransform":
            from music_genre_classification.train_data_transforms import (
                ResamplerDataTransform,
            )

            return ResamplerDataTransform(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
