import itertools
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.optim import Adam
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
        ewc_lambda: float = 0.1,
        optimizer_config: dict = {"lr": 0.001, "weight_decay": 2e-4, "momentum": 0.9},
        mode: str = "separate",
    ):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.ewc_lambda = ewc_lambda

        self.keep_importance_data = True

        self.saved_params: dict[int, dict[str, dict]] = defaultdict(dict)
        self.importances: dict[int, dict[str, dict]] = defaultdict(dict)

        self.optimizer_config = optimizer_config
        self.mode = mode

    def configure(self, parameters: Iterable[Tensor] | Iterable[dict]):
        self.optimizer = Adam(parameters, **self.optimizer_config)

    def before_backward(self, model: nn.Module, task_id: int):
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
                if imp.shape != new_shape:
                    # Only for last fc layer
                    penalty += (
                        imp * (cur_param[: imp.shape[0]] - saved_param).pow(2)
                    ).sum()
                else:
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()

        return self.ewc_lambda * penalty

    def after_training_task(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        data_transform: nn.Module,
        criteria: Criteria,
        task_id: int,
    ):
        """
        Compute importances of parameters after each experience.
        """
        logger.info(f"Computing importances for task {task_id}")
        importances = self.compute_importances(
            model,
            criteria,
            dataloader,
            data_transform,
        )
        self.update_importances(importances, task_id)
        self.saved_params[task_id] = self.copy_params_dict(model)
        # clear previous parameter values
        if task_id > 0 and (not self.keep_importance_data):
            del self.saved_params[task_id - 1]

    def compute_importances(
        self,
        model: nn.Module,
        criteria: Criteria,
        dataloader: DataLoader,
        data_transform: nn.Module,
    ) -> dict[str, dict]:
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # list of list
        importances = self.zeroslike_params_dict(model)
        for x, y in dataloader:
            x = x.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)

            x = data_transform(x)

            self.optimizer.zero_grad()
            out = model(x)
            loss = criteria(out, y)
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
    def update_importances(self, importances: dict, task_id: int):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or task_id == 0:
            self.importances[task_id] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                self.importances[task_id - 1].items(),
                importances.items(),
                fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    assert k2 is not None
                    assert curr_imp is not None
                    self.importances[task_id][k2] = curr_imp
                    continue

                assert k1 == k2, "Error in importance computation."
                assert curr_imp is not None
                assert old_imp is not None
                assert k2 is not None

                # manage expansion of existing layers
                self.importances[task_id][k1] = dict(
                    f"imp_{k1}",
                    curr_imp.shape,
                    init_tensor=self.decay_factor * old_imp.expand(curr_imp.shape)
                    + curr_imp.data,
                    device=curr_imp.device,
                )

            # clear previous parameter importances
            if task_id > 0 and (not self.keep_importance_data):
                del self.importances[task_id - 1]

        else:
            raise ValueError("Wrong EWC mode.")

    @staticmethod
    def zeroslike_params_dict(model: nn.Module) -> dict[str, dict]:
        return dict(
            [
                (k, torch.zeros_like(p, dtype=p.dtype, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    @staticmethod
    def copy_params_dict(model: nn.Module) -> dict[str, dict]:
        return dict([(k, p.clone()) for k, p in model.named_parameters()])
