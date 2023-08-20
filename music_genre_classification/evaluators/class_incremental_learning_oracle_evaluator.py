from loguru import logger

import config
from music_genre_classification.evaluators.class_incremental_learning_evaluator import (
    ClassIncrementalLearningEvaluator,
)


class ClassIncrementalLearningOracleEvaluator(ClassIncrementalLearningEvaluator):
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

        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            task_id=0,
            task="all",
        )
        self.model_saver.load_model()
        self.model.to(config.device)
        self.data_transform.to(config.device)

    def configure_task(self, task_id: int, task: list[str] | str):
        self.data_transform.to(config.device)

        self.experiment_tracker.configure_task(
            train_task_number=task_id,
            train_task_name=task,
        )

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

        accumulated_tasks = []
        for task_id, task in enumerate(self.tasks):
            accumulated_tasks += task
            logger.info(f"Started evaluation of {accumulated_tasks=}")
            self.configure_task(task_id=task_id, task=task)
            data_loader = self.data_source.get_dataset(
                tasks=["all"], task=accumulated_tasks
            )
            results = self.predict(data_loader)
            metrics = self.extract_metrics(results)
            self.experiment_tracker.log_task_metrics(metrics, accumulated_tasks)
