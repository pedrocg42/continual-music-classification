from loguru import logger

import config
from music_genre_classification.evaluators.evaluator import Evaluator
from music_genre_classification.metrics import MetricsFactory


class ClassIncrementalLearningEvaluator(Evaluator):
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

        # Updating metrics
        for metric_config in self.metrics_config:
            metric_config["args"].update({"num_classes": self.model.num_classes})

        self.metrics = MetricsFactory.build(self.metrics_config)

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

            accumulated_tasks = []
            for task_id, task in enumerate(self.tasks):
                accumulated_tasks += task
                logger.info(f"Started evaluation of model train with {task=}")
                self.configure_task(
                    cross_val_id=cross_val_id, task_id=task_id, task=task
                )

                logger.info(f"Started evaluation of {accumulated_tasks=}")
                data_loader = self.data_source.get_dataset(
                    cross_val_id=cross_val_id, task=accumulated_tasks
                )
                results = self.predict(data_loader)
                metrics = self.extract_metrics(results)
                self.experiment_tracker.log_task_metrics(metrics, accumulated_tasks)
