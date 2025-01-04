from src.loopers.dkvb_music_genre_classification_looper import (
    DkvbMusicGenreClassificationLooper,
)
from src.loopers.ewc_music_genre_classification_looper import (
    EwcMusicGenreClassificationLooper,
)
from src.loopers.gem_music_genre_classification_looper import (
    GemMusicGenreClassificationLooper,
)
from src.loopers.icarl_music_genre_classification_looper import (
    iCaRLMusicGenreClassificationLooper,
)
from src.loopers.l2p_music_genre_classification_looper import (
    L2PMusicGenreClassificationLooper,
)
from src.loopers.looper import Looper
from src.loopers.looper_factory import LooperFactory
from src.loopers.music_continual_learning_embedding_looper import (
    MusicContinualLearningEmbeddingLooper,
)
from src.loopers.music_genre_classification_looper import (
    MusicGenreClassificationLooper,
)

__all__ = [
    "Looper",
    "LooperFactory",
    "MusicGenreClassificationLooper",
    "DkvbMusicGenreClassificationLooper",
    "GemMusicGenreClassificationLooper",
    "EwcMusicGenreClassificationLooper",
    "L2PMusicGenreClassificationLooper",
    "iCaRLMusicGenreClassificationLooper",
    "MusicContinualLearningEmbeddingLooper",
]
