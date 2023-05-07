import torch
from loguru import logger
from tqdm import tqdm

import config
from music_genre_classification.evaluators import Evaluator


class ContinualLearningEvaluator(Evaluator):
    def __init__(
        self,
        train_tasks: list[str],
        test_tasks: list[str],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

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

    def configure_task(self, cross_val_id: int, task_num: int, task: str):
        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            cross_val_id=cross_val_id,
            task=task,
        )
        self.model_saver.load_model()
        self.model.to(config.device)
        self.experiment_tracker.configure_task(
            cross_val_id=cross_val_id,
            train_task_number=task_num,
            train_task_name=task,
        )

    def predict(self, data_loader) -> list[dict]:
        self.model.eval()
        results = []
        for i, (waveforms, labels) in enumerate(tqdm(data_loader, colour="green")):
            if self.debug and i == self.max_steps:
                break

            waveforms = waveforms.to(config.device)

            # Inference
            transformed = self.data_transform(waveforms)
            preds = self.model(transformed)

            # For each song we select the most repeated class
            pred = preds.detach().cpu().mean(dim=1)
            label = labels[0] if len(labels.shape) > 0 else labels

            results.append(
                dict(
                    pred=pred,
                    label=label,
                )
            )
        return results

    def extract_metrics(self, results: list[dict]) -> dict:
        metrics = {}
        preds = torch.vstack([result["pred"] for result in results])
        labels = torch.hstack([result["label"] for result in results])
        for metric_name, metric in self.metrics.items():
            metrics[metric_name] = metric(preds, labels).item()
        return metrics

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
            if self.debug and cross_val_id > 0:
                break
            for task_num, task in enumerate(self.train_tasks):
                logger.info(f"Started evaluation of task {task}")
                self.configure_task(
                    cross_val_id=cross_val_id, task_num=task_num, task=task
                )
                # Extracting results for all tasks
                data_loader = self.data_source.get_dataset(cross_val_id=cross_val_id)
                results = self.predict(data_loader)
                metrics = self.extract_metrics(results)
                self.experiment_tracker.log_task_metrics(metrics, "all")
                # Extracting results per task
                for task_test in self.test_tasks:
                    data_loader = self.data_source.get_dataset(
                        cross_val_id=cross_val_id, task=task_test
                    )
                    results = self.predict(data_loader)
                    metrics = self.extract_metrics(results)
                    self.experiment_tracker.log_task_metrics(metrics, task_test)
