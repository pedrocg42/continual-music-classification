from music_genre_classification.criterias.criteria import Criteria
from music_genre_classification.criterias.torch_cross_entropy_criteria import (
    TorchCrossEntropyCriteria,
)
from music_genre_classification.criterias.criteria_factory import CriteriaFactory

__all__ = ["CriteriaFactory", "Criteria", "TorchCrossEntropyCriteria"]
