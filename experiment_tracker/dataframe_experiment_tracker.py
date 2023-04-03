import os

import pandas as pd
from loguru import logger

import config
from experiment_tracker.experiment_tracker import ExperimentTracker


class DataframeExperimentTracker(ExperimentTracker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.dataframe_path = os.path.join(config.results_path, "results.csv")
        self.dataframe = (
            pd.read_csv(self.dataframe_path)
            if os.path.exists(self.dataframe_path)
            else pd.DataFrame()
        )

    def configure(
        self,
        experiment_name: str,
        experiment_type: str,
        experiment_subtype: str,
        dataset_name: str,
    ):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.experiment_subtype = experiment_subtype
        self.dataset_name = dataset_name

    def configure_task(
        self,
        cross_val_id: int,
        train_task_number: int = 0,
        train_task_name: str = "all",
        task_number: int = 0,
    ):
        self.cross_val_id = cross_val_id
        self.train_task_number = train_task_number
        self.train_task_name = train_task_name
        self.task_number = task_number

        self.row = {
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "experiment_subtype": self.experiment_subtype,
            "cv_fold": self.cross_val_id,
            "train_dataset_name": self.dataset_name,
            "train_task_number": self.train_task_number,
            "train_task_name": self.train_task_name,
        }

    def log_metrics(
        self,
        metrics: str,
        task: str = None,
    ):
        row = self.row.copy()
        row["task_name"] = task
        for metric_name, metric_value in metrics.items():
            row[metric_name] = metric_value.numpy()
        row = pd.Series(row)

        self.add_row(row)
        self.save_results()

    def log_tasks_metrics(
        self,
        metrics: dict[str, list[float]],
        tasks: list[str],
    ):
        row = self.row.copy()
        row["task_name"] = "all"
        # Extracting average metrics
        for metric_name, metric_values in metrics.items():
            row[metric_name] = sum(metric_values) / len(metric_values)
        row = pd.Series(row)
        self.add_row(row)

        # Extracting individual metrics
        for task_number, task_name in enumerate(tasks):
            row = self.row.copy()
            row["task_name"] = task_name
            for metric_name, metric_values in metrics.items():
                row[metric_name] = metric_values[task_number]
            row = pd.Series(row)
            self.add_row(row)

        self.save_results()

    def add_row(self, row: dict):
        row = pd.Series(row)
        if len(self.dataframe) == 0:
            self.dataframe = pd.DataFrame([row])
        else:
            self.dataframe = pd.concat(
                [
                    self.dataframe,
                    pd.DataFrame([row]),
                ],
                ignore_index=True,
            )

    def save_results(self):
        logger.info(f"Saving results to {self.dataframe_path}")
        self.dataframe.to_csv(self.dataframe_path, index=False)
