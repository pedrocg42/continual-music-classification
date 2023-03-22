from fire import Fire

from trainers import MusicGenderClassificationTrainer
from utils import parse_experiment


@parse_experiment
def train(
    **experiment,
):
    trainer = MusicGenderClassificationTrainer(**experiment)
    trainer.train()


if __name__ == "__main__":
    Fire(train)
