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

    def log_metrics(
        self,
        metrics: str,
        cv_fold: int = 0,
        task_number: int = 0,
        task_name: str = "",
        train_subdataset_name: str = "",
    ):
        row = {
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "experiment_subtype": self.experiment_subtype,
            "train_dataset_name": self.dataset_name,
            "train_subdataset_name": train_subdataset_name,
            "cv_fold": cv_fold,
            "task_number": task_number,
            "task_name": task_name,
        }
        for metric_name, metric_value in metrics.items():
            row[metric_name] = metric_value.numpy()
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