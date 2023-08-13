from fire import Fire

from music_genre_classification.evaluators import EvaluatorFactory
from music_genre_classification.my_utils import parse_experiment
from music_genre_classification.trainers import TrainerFactory


def train(experiment_name: str, trainer: dict):
    trainer = TrainerFactory.build(trainer)
    trainer.train(experiment_name)


def evaluate(
    experiment_name: str,
    experiment_type: str,
    experiment_subtype: str,
    evaluator: dict,
):
    evaluator = EvaluatorFactory.build(evaluator)
    evaluator.evaluate(experiment_name, experiment_type, experiment_subtype)


@parse_experiment
def execute_experiment(
    experiment_name: str,
    experiment_type: str,
    experiment_subtype: str,
    **experiment,
):
    train(experiment_name, **experiment["train"])
    evaluate(
        experiment_name,
        experiment_type,
        experiment_subtype,
        **experiment["evaluate"],
    )

    with open("finished_experiments.txt", "a") as f:
        f.write(experiment_name)


if __name__ == "__main__":
    Fire(execute_experiment)
