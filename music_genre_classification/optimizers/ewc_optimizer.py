from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader

import config
from music_genre_classification.criterias.criteria import Criteria
from music_genre_classification.optimizers.torch_base_optimizer import (
    TorchBaseOptimizer,
)


class EwcOptimizer(TorchBaseOptimizer):
    """
    Gradient Episodic Memory
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(
        self,
        ewc_lambda,
        decay_factor=None,
        optimizer_kwargs: dict = {},
    ):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.ewc_lambda = ewc_lambda
        self.decay_factor = decay_factor

        self.keep_importance_data = True

        self.saved_params: Dict[int, Dict[str, ParamData]] = defaultdict(dict)
        self.importances: Dict[int, Dict[str, ParamData]] = defaultdict(dict)

        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = SGD

    def configure(self, parameters: Iterable[Tensor] | Iterable[dict], **kwargs):
        self.optimizer = AdamW(parameters, **self.optimizer_kwargs)

    def before_backward(self, model: nn.Module, task_id: int, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """
        if task_id == 0:
            return

        penalty = torch.tensor(0).float().to(config.device)

        for experience in range(task_id):
            for k, cur_param in model.named_parameters():
                # new parameters do not count
                if k not in self.saved_params[experience]:
                    continue
                saved_param = self.saved_params[experience][k]
                imp = self.importances[experience][k]
                new_shape = cur_param.shape
                penalty += (
                    imp.expand(new_shape)
                    * (cur_param - saved_param.expand(new_shape)).pow(2)
                ).sum()

        return self.ewc_lambda * penalty

    def after_training_task(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        task_id: int,
        criteria: Criteria,
        batch_size: int,
        **kwargs,
    ):
        """
        Compute importances of parameters after each experience.
        """
        importances = self.compute_importances(
            model,
            criteria,
            dataloader,
            config.device,
            batch_size,
        )
        self.update_importances(importances, task_id)
        self.saved_params[task_id] = copy_params_dict(strategy.model)
        # clear previous parameter values
        if task_id > 0 and (not self.keep_importance_data):
            del self.saved_params[task_id - 1]

    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size
    ) -> Dict[str, ParamData]:
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        if device == "cuda":
            for module in model.modules():
                module.train()

        # list of list
        importances = zerolike_params_dict(model)
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t: int):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                self.importances[t - 1].items(),
                importances.items(),
                fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    assert k2 is not None
                    assert curr_imp is not None
                    self.importances[t][k2] = curr_imp
                    continue

                assert k1 == k2, "Error in importance computation."
                assert curr_imp is not None
                assert old_imp is not None
                assert k2 is not None

                # manage expansion of existing layers
                self.importances[t][k1] = ParamData(
                    f"imp_{k1}",
                    curr_imp.shape,
                    init_tensor=self.decay_factor * old_imp.expand(curr_imp.shape)
                    + curr_imp.data,
                    device=curr_imp.device,
                )

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")
