from fire import Fire

from src.evaluators import EvaluatorFactory
from src.my_utils import parse_experiment
from src.trainers import TrainerFactory


def train(experiment_name: str, trainer: dict):
    trainer["args"]["debug"] = True
    trainer = TrainerFactory.build(trainer)
    trainer.train(experiment_name)


def evaluate(
    experiment_name: str,
    experiment_type: str,
    experiment_subtype: str,
    evaluator: dict,
):
    evaluator["args"]["debug"] = True
    evaluator = EvaluatorFactory.build(evaluator)
    evaluator.evaluate(experiment_name, experiment_type, experiment_subtype)


@parse_experiment
def execute_experiment_check(
    experiment_name: str,
    experiment_type: str,
    experiment_subtype: str,
    **experiment,
):
    # try:
    train(experiment_name, **experiment["train"])
    evaluate(
        experiment_name,
        experiment_type,
        experiment_subtype,
        **experiment["evaluate"],
    )
    with open("finished_experiments.txt", "a") as f:
        f.write(f"{experiment_name}\n")
    # except Exception as e:
    #     with open("errors.txt", "a") as f:
    #         f.write("-" * 15 + f"{experiment_name}" + "-" * 15 + "\n")
    #         f.write(str(e))
    #         f.write(traceback.format_exc())


if __name__ == "__main__":
    Fire(execute_experiment_check)
