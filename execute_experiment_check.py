from fire import Fire

from music_genre_classification.my_utils import parse_experiment
from music_genre_classification.trainers import TrainerFactory
from music_genre_classification.evaluators import EvaluatorFactory


def train(experiment_name: str, num_cross_val_splits: int, trainer: dict):
    trainer["args"]["debug"] = True
    trainer = TrainerFactory.build(trainer)
    trainer.train(experiment_name, num_cross_val_splits)


def evaluate(
    experiment_name: str,
    experiment_type: str,
    experiment_subtype: str,
    num_cross_val_splits: int,
    evaluator: dict,
):
    evaluator["args"]["debug"] = True
    evaluator = EvaluatorFactory.build(evaluator)
    evaluator.evaluate(
        experiment_name, experiment_type, experiment_subtype, num_cross_val_splits
    )


@parse_experiment
def execute_experiment_check(
    experiment_name: str,
    experiment_type: str,
    experiment_subtype: str,
    num_cross_val_splits: int,
    **experiment,
):
    train(experiment_name, num_cross_val_splits, **experiment["train"])
    evaluate(
        experiment_name,
        experiment_type,
        experiment_subtype,
        num_cross_val_splits,
        **experiment["evaluate"],
    )


if __name__ == "__main__":
    Fire(execute_experiment_check)
