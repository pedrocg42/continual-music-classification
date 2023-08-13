from loguru import logger

import config
from music_genre_classification.evaluators.evaluator import Evaluator


class ClassIncrementalLearningOracleEvaluator(Evaluator):
    def __init__(
        self,
        tasks: list[str | list[str]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tasks = tasks

    def configure(
        self,
        experiment_name: str,
        experiment_type: str,
        experiment_subtype: str,
    ):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.experiment_subtype = experiment_subtype

        self.experiment_tracker.configure(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            experiment_subtype=experiment_subtype,
            dataset_name=self.data_source.name,
        )
        self.experiment_tracker.configure_task(
            train_task_number=0,
            train_task_name="all",
        )

        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            task_id=0,
            task="all",
        )
        self.model_saver.load_model()
        self.model.to(config.device)
        self.data_transform.to(config.device)

    def evaluate(
        self,
        experiment_name: str,
        experiment_type: str,
        experiment_subtype: str,
    ):
        logger.info(f"Started evaluation process of experiment {experiment_name}")
        self.configure(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            experiment_subtype=experiment_subtype,
        )

        # Extracting results for all tasks
        logger.info("Started evaluation of all tasks")
        data_loader = self.data_source.get_dataset(tasks=self.tasks, task="all")
        results = self.predict(data_loader)
        metrics = self.extract_metrics(results)
        self.experiment_tracker.log_task_metrics(metrics, "all")
