from music_genre_classification.loopers.looper import Looper


class MusicGenreClassificationLooper(Looper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_id = None
        self.task = None
