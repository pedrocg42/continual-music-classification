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
        self, experiment_name: str, experiment_type: str, experiment_subtype: str
    ):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.experiment_subtype = experiment_subtype

    def log_metrics(
        self, metrics: str, cv_fold: int = 0, task_number: int = 0, task_name: str = ""
    ):
        row = {
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "experiment_subtype": self.experiment_subtype,
            "cv_fold": cv_fold,
            "task_number": task_number,
            "task_name": task_name,
        }
        for metric_name, metric_value in metrics.items():
            row[metric_name] = metric_value

        if len(self.dataframe) == 0:
            self.dataframe = pd.DataFrame(row)
        else:
            self.dataframe = pd.concat(
                [
                    self.dataframe,
                    pd.DataFrame([row]),
                ],
                ignore_index=True,
            )

        self.save_results()

    def save_results(self):
        logger.info(f"Saving results to {self.dataframe_path}")
        self.dataframe.to_csv(self.dataframe_path, index=False)
