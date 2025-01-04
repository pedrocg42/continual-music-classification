from src.trainers.replay_continual_learning_trainer import (
    ReplayContinualLearningTrainer,
)


class iCaRLContinualLearningTrainer(ReplayContinualLearningTrainer):
    def after_training_task(self, task: list[str] | str):
        super().after_training_task(task)
        self.looper.old_model = self.model.copy().freeze()
        self.looper.known_classes = self.known_classes
