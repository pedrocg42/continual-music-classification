import torch
from loguru import logger
from tqdm import tqdm

import config
from src.experiment_trackers import ExperimentTrackerFactory
from src.metrics import MetricsFactory
from src.model_savers import ModelSaverFactory
from src.models import TrainModelFactory
from src.train_data_sources import TrainDataSourceFactory
from src.train_data_transforms import TrainDataTransformFactory


class Evaluator:
    def __init__(
        self,
        model: dict,
        model_saver: dict,
        data_source: dict,
        data_transform: dict,
        metrics_config: list[dict],
        experiment_tracker: dict,
        debug: bool = False,
    ):
        # Basic information
        self.experiment_name = None
        self.experiment_type = None
        self.experiment_subtype = None
        self.metrics_config = metrics_config

        # Components
        self.model = TrainModelFactory.build(model)
        self.model_saver = ModelSaverFactory.build(model_saver)
        self.data_source = TrainDataSourceFactory.build(data_source)
        self.data_transform = TrainDataTransformFactory.build(data_transform)
        self.metrics = MetricsFactory.build(self.metrics_config)
        self.experiment_tracker = ExperimentTrackerFactory.build(experiment_tracker)

        # Debug
        self.debug = debug
        self.max_steps = 5

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
        self.model_saver.configure(self.model, experiment_name=experiment_name)
        self.model_saver.load_model()
        self.model.to(config.device)

    @torch.no_grad()
    def predict(self, data_loader) -> list[dict]:
        self.model.eval()
        results = []
        pbar = tqdm(
            data_loader,
            colour="green",
            total=self.max_steps if self.debug else len(data_loader),
        )
        for i, (waveforms, labels) in enumerate(pbar):
            if self.debug and i == self.max_steps:
                break

            waveforms = waveforms.to(config.device)

            # Inference
            transformed = self.data_transform(waveforms)
            preds = self.model(transformed)

            # For each song we select the most repeated class
            pred = preds.detach().cpu().mean(dim=0).softmax(dim=0)
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
            metric.num_classes = self.model.num_classes
            metrics[metric_name] = metric(preds, labels).item()
        return metrics

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

        data_loader = self.data_source.get_dataset(tasks=self.tasks, task="all")
        results = self.predict(data_loader)
        metrics = self.extract_metrics(results)
        self.experiment_tracker.log_metrics(metrics)
