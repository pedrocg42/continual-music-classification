from fire import Fire

from evaluators import Evaluator
from my_utils import parse_experiment
from trainers import Trainer


def train(experiment_name: str, trainer: Trainer):
    trainer.train(experiment_name)


def evaluate(
    experiment_name: str,
    experiment_type: str,
    experiment_subtype: str,
    evaluator: Evaluator,
):
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
        experiment_name, experiment_type, experiment_subtype, **experiment["evaluate"]
    )


if __name__ == "__main__":
    Fire(execute_experiment)
