import torch
from tqdm import tqdm

import config
from src.evaluators.class_incremental_learning_evaluator import (
    ClassIncrementalLearningEvaluator,
)


class ClassIncrementalLearningL2PEvaluator(ClassIncrementalLearningEvaluator):
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
            preds, _ = self.model(transformed)

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
