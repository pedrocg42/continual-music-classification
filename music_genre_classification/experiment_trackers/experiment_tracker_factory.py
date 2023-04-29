from music_genre_classification.experiment_trackers.experiment_tracker import (
    ExperimentTracker,
)


class ExperimentTrackerFactory:
    """
    Factory class for creating experiment trackers.
    """

    def build(config: dict) -> ExperimentTracker:
        if config["name"] == "TensorboardExperimentTracker":
            from music_genre_classification.experiment_trackers import (
                TensorboardExperimentTracker,
            )

            return TensorboardExperimentTracker(**config.get("args", {}))
        elif config["name"] == "DataframeExperimentTracker":
            from music_genre_classification.experiment_trackers import (
                DataframeExperimentTracker,
            )

            return DataframeExperimentTracker(**config.get("args", {}))
        else:
            raise ValueError(f"Unknown TrainDataSource type: {config['name']}")
