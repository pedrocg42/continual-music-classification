from loguru import logger

import config
from music_genre_classification.evaluators import Evaluator


class ContinualLearningEvaluator(Evaluator):
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
        num_cross_val_splits: int,
    ):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.experiment_subtype = experiment_subtype
        self.num_cross_val_splits = num_cross_val_splits

        self.experiment_tracker.configure(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            experiment_subtype=experiment_subtype,
            dataset_name=self.data_source.name,
        )

    def configure_task(self, cross_val_id: int, task_id: int, task: str):
        self.model.update_decoder(task_id, task)
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            cross_val_id=cross_val_id,
            task_id=task_id,
            task=task,
        )
        self.model_saver.load_model()
        self.model.to(config.device)
        self.data_transform.to(config.device)
        self.experiment_tracker.configure_task(
            cross_val_id=cross_val_id,
            train_task_number=task_id,
            train_task_name=task,
        )

    def evaluate(
        self,
        experiment_name: str,
        experiment_type: str,
        experiment_subtype: str,
        num_cross_val_splits: int,
    ):
        logger.info(f"Started evaluation process of experiment {experiment_name}")
        self.configure(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            experiment_subtype=experiment_subtype,
            num_cross_val_splits=num_cross_val_splits,
        )

        for cross_val_id in range(self.num_cross_val_splits):
            if cross_val_id > 0 or self.debug and cross_val_id > 0:
                break
            logger.info(f"Started evaluation of cross-validation {cross_val_id=}")
            for task_id, task in enumerate(self.tasks):
                logger.info(f"Started evaluation of model train with {task=}")
                self.configure_task(
                    cross_val_id=cross_val_id, task_id=task_id, task=task
                )
                # Extracting results per task
                for test_task in self.tasks:
                    logger.info(f"Started evaluation of {test_task=}")
                    data_loader = self.data_source.get_dataset(
                        cross_val_id=cross_val_id, task=test_task
                    )
                    results = self.predict(data_loader)
                    metrics = self.extract_metrics(results)
                    self.experiment_tracker.log_task_metrics(metrics, test_task)

            # Extracting results for all tasks
            logger.info("Started evaluation of all tasks")
            data_loader = self.data_source.get_dataset(
                cross_val_id=cross_val_id, task="all"
            )
            results = self.predict(data_loader)
            metrics = self.extract_metrics(results)
            self.experiment_tracker.log_task_metrics(metrics, test_task)
