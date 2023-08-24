from typing import Dict, Iterable

import numpy as np
import quadprog
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
from music_genre_classification.train_data_transforms import TrainDataTransform


class GemOptimizer(TorchBaseOptimizer):
    """
    Gradient Episodic Memory
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of memories from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(
        self,
        num_memories: int,
        memory_strength: float,
        optimizer_config: dict = {"lr": 0.001},
    ):
        """
        :param memories_per_class: number of memories per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.num_memories = num_memories
        self.memories_per_class = None
        self.memory_strength = memory_strength
        self.optimizer_config = optimizer_config

        self.memory_x: Dict[int, Tensor] = dict()
        self.memory_y: Dict[int, Tensor] = dict()

        self.G: Tensor = torch.empty(0)

        self.known_classes = []

    def configure(self, parameters: Iterable[Tensor] | Iterable[dict]):
        self.optimizer = Adam(parameters, **self.optimizer_config)

    def before_training_iteration(
        self,
        model: nn.Module,
        criteria: Criteria,
        data_transform: TrainDataTransform,
        **kwargs,
    ):
        """
        Compute gradient constraints on previous memory samples from all
        experiences.
        """

        if len(self.known_classes) > 0:
            model.prepare_train()
            G = []
            for i in range(len(self.known_classes)):
                self.optimizer.zero_grad()
                xref = data_transform(
                    self.memory_x[i].to(config.device, non_blocking=True)
                )
                yref = self.memory_y[i].to(config.device, non_blocking=True)
                out = model(xref)
                loss = criteria(out, yref)
                loss.backward()

                temp_grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        temp_grads.append(p.grad.flatten().detach().clone())
                G.append(torch.cat(temp_grads, dim=0))

            self.G = torch.stack(G)  # (experiences, parameters)

    @torch.no_grad()
    def after_backward(self, model: nn.Module, task_id: int, **kwargs):
        """
        Project gradient based on reference gradients and
        """

        if task_id > 0:
            temp_grads = []
            for p in model.parameters():
                if p.grad is not None:
                    temp_grads.append(p.grad.flatten().detach().clone())
            g = torch.cat(temp_grads, dim=0)

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in model.parameters():
                if p.grad is not None:
                    curr_pars = p.numel()
                    p.grad.copy_(
                        v_star[num_pars : num_pars + curr_pars]
                        .view(p.size())
                        .to(config.device)
                    )
                    num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        task_id = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(task_id) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(task_id)
        h = np.zeros(task_id) + self.memory_strength
        v = quadprog.solve_qp(P, q, G, h)[0]
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()

    def after_training_task(
        self,
        task: list[str],
        dataloader: DataLoader,
    ):
        self.memories_per_class = self.num_memories // len(self.known_classes + task)
        if len(self.known_classes):
            self.reduce_exemplar(task)
        self.known_classes += task
        self.update_memory(dataloader, task)

    def reduce_exemplar(self, task):
        logger.info(f"Reducing memories...({self.memories_per_class} per classes)")
        for task_id in self.memory_x.keys():
            self.memory_x[task_id] = self.memory_x[task_id][: self.memories_per_class]
            self.memory_y[task_id] = self.memory_y[task_id][: self.memories_per_class]

    @torch.no_grad()
    def update_memory(self, dataloader: DataLoader, task: list[str]):
        """
        Update memory with memories from current experience.
        """
        logger.info(f"Update memories...({self.memories_per_class} per class)")
        class_ids = []
        for class_name in task:
            class_id = self.known_classes.index(class_name)
            class_ids.append(class_id)
            self.memory_x[class_id] = []
            self.memory_y[class_id] = []

        for mbatch in dataloader:
            x, y = mbatch[0], mbatch[1]
            for class_id in class_ids:
                mask = y == class_id
                if len(self.memory_x[class_id]) < self.memories_per_class and torch.any(
                    mask
                ):
                    x_class = x[mask]
                    y_class = y[mask]
                    if (
                        len(self.memory_x[class_id]) + len(x_class)
                        <= self.memories_per_class
                    ):
                        if len(self.memory_x[class_id]) == 0:
                            self.memory_x[class_id] = x_class.clone()
                            self.memory_y[class_id] = y_class.clone()
                        else:
                            self.memory_x[class_id] = torch.cat(
                                (self.memory_x[class_id], x_class), dim=0
                            )
                            self.memory_y[class_id] = torch.cat(
                                (self.memory_y[class_id], y_class), dim=0
                            )
                    else:
                        diff = self.memories_per_class - len(self.memory_x[class_id])
                        if len(self.memory_x[class_id]) == 0:
                            self.memory_x[class_id] = x_class[:diff].clone()
                            self.memory_y[class_id] = y_class[:diff].clone()
                        else:
                            self.memory_x[class_id] = torch.cat(
                                (self.memory_x[class_id], x_class[:diff]), dim=0
                            )
                            self.memory_y[class_id] = torch.cat(
                                (self.memory_y[class_id], y_class[:diff]), dim=0
                            )
            if all(
                [
                    len(self.memory_x[class_id]) >= self.memories_per_class
                    for class_id in class_ids
                ]
            ):
                break

        for class_id in class_ids:
            if self.memory_x[class_id].size(0) < self.memories_per_class:
                logger.warning(
                    f"Unable to find {self.memories_per_class} for class {self.known_classes[class_id]} - Collected {self.memory_x[class_id].size(0)} memories"
                )
