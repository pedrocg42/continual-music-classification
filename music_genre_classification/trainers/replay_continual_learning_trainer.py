import copy

import numpy as np
import torch
from loguru import logger

import config
from music_genre_classification.train_data_sources.memory_dataset import MemoryDataset
from music_genre_classification.trainers.class_incremental_learning_trainer import (
    ClassIncrementalLearningTrainer,
)


class ReplayContinualLearningTrainer(ClassIncrementalLearningTrainer):
    def __init__(self, num_memories: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_memories = num_memories
        self.known_classes = []
        self.data_memory = np.array([])
        self.targets_memory = np.array([])
        self.memory_dataset = None

    def configure_task(
        self, cross_val_id: int, task_id: int, task: list[str], **kwargs
    ):
        super().configure_task(cross_val_id, task_id, task, **kwargs)
        self.memories_per_class = self.num_memories // len(self.known_classes + task)

        if self.memory_dataset is not None:
            self.train_data_loader = self.train_data_source.get_dataloader(
                cross_val_id=cross_val_id,
                task=task,
                batch_size=self.batch_size,
                memory_dataset=self.memory_dataset,
            )

    def train(self, experiment_name: str, num_cross_val_splits: int = 1):
        logger.info(f"Started training process of experiment {experiment_name}")
        self.configure_experiment(experiment_name, self.batch_size)
        for cross_val_id in range(num_cross_val_splits):
            if cross_val_id > 0 or self.debug and cross_val_id > 0:
                break
            self.configure_cv(cross_val_id)
            self.log_start()
            for task_id, task in enumerate(self.tasks):
                self.configure_task(cross_val_id, task_id, task)
                if self.model_saver.model_exists():
                    logger.info(
                        f"Model already exists for cross_val_id {cross_val_id} and task {task}"
                    )
                    self.after_training_task(cross_val_id, task)
                    continue
                logger.info(f"Starting training of task {task}")
                for epoch in range(self.num_epochs):
                    early_stopping = self.train_epoch(epoch)
                    if early_stopping:
                        break
                self.after_training_task(cross_val_id, task)

    def after_training_task(self, cross_val_id: int, task: list[str]):
        if len(self.known_classes):
            self.reduce_exemplar()
        self.known_classes += task
        self.construct_exemplar(cross_val_id, task)

    def reduce_exemplar(self):
        logger.info(f"Reducing exemplars...({self.memories_per_class} per classes)")
        dummy_data = copy.deepcopy(self.data_memory)
        dummy_targets = copy.deepcopy(self.targets_memory)
        self.data_memory, self.targets_memory = np.array([]), np.array([])

        for class_idx in range(len(self.known_classes)):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = (
                dummy_data[mask][: self.memories_per_class],
                dummy_targets[mask][: self.memories_per_class],
            )
            self.data_memory = (
                np.concatenate((self.data_memory, dd))
                if len(self.data_memory) != 0
                else dd
            )
            self.targets_memory = (
                np.concatenate((self.targets_memory, dt))
                if len(self.targets_memory) != 0
                else dt
            )

    def construct_exemplar(self, cross_val_id: int, task: str | list[str]):
        logger.info(f"Constructing exemplars...({self.memories_per_class} per class)")
        for class_idx, task_class in enumerate(task):
            inputs, _, vectors = self.extract_vectors(cross_val_id, task_class)
            vectors = (
                vectors
                / (np.linalg.norm(vectors.T, axis=0) + np.finfo(np.float32).eps)[
                    :, np.newaxis
                ]
            )

            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, self.memories_per_class + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(inputs[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                inputs = np.delete(
                    inputs, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(self.memories_per_class, class_idx)
            self.data_memory = (
                np.concatenate((self.data_memory, selected_exemplars))
                if len(self.data_memory) != 0
                else selected_exemplars
            )
            self.targets_memory = (
                np.concatenate((self.targets_memory, exemplar_targets))
                if len(self.targets_memory) != 0
                else exemplar_targets
            )

            self.memory_dataset = MemoryDataset(
                data=self.data_memory, targets=self.targets_memory
            )

    @torch.no_grad()
    def extract_vectors(self, cross_val_id: int, task: str):
        self.model.eval()

        # Getting data loader
        data_loader = self.train_data_source.get_dataset(
            cross_val_id=cross_val_id, task=task, is_eval=True
        )

        inputs_list, targets_list, vectors_list = [], [], []
        for inputs, targets in data_loader:
            transformed_inputs = self.train_data_transform(inputs.to(config.device))
            vectors = self.model.extract_vector(transformed_inputs)
            inputs_list.append(inputs.cpu().detach().numpy())
            targets_list.append(targets.detach().numpy())
            vectors_list.append(vectors.cpu().detach().numpy())
        return (
            np.concatenate(inputs_list),
            np.concatenate(targets_list),
            np.concatenate(vectors_list),
        )
