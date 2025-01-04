import torch
from tqdm import tqdm

import config
from src.evaluators.class_incremental_learning_evaluator import (
    ClassIncrementalLearningEvaluator,
)
from src.metrics import MetricsFactory


class ClassIncrementalLearningL2CenterEvaluator(ClassIncrementalLearningEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = 0

    def configure_task(self, task_id: int, task: list[str] | str):
        self.num_classes += len(task)

        self.model.to(config.device)
        self.model.reference_embeddings = torch.zeros(
            (self.num_classes, self.model.encoder.output_size)
        )

        self.model_saver.configure(
            self.model,
            experiment_name=self.experiment_name,
            task_id=task_id,
            task=task,
        )
        self.model_saver.load_model()

        self.data_transform.to(config.device)
        self.experiment_tracker.configure_task(
            train_task_number=task_id,
            train_task_name=task,
        )

        # Updating metrics
        for metric_config in self.metrics_config:
            metric_config["args"].update({"num_classes": self.model.num_classes})
        self.metrics = MetricsFactory.build(self.metrics_config)

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
            embeddings = self.model.forward_features(transformed).detach().cpu()

            # Average embeddings
            mean_embedding = embeddings.mean(dim=0, keepdim=True)
            preds = self.model.match_embeddings(mean_embedding)

            # For each song we select the most repeated class
            preds = preds.softmax(dim=1)
            label = labels[0] if len(labels.shape) > 0 else labels

            results.append(
                dict(
                    pred=preds,
                    label=label,
                )
            )
        return results
