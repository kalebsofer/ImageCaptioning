import random


class NoteGenerator:
    def __init__(self):
        self.notes = ["A", "B", "C", "D", "E", "F", "G"]
        self.genres = ["rock", "jazz", "classical", "pop", "blues"]
        self.genre_sequences = {genre: [] for genre in self.genres}

    def set_genre(self, genre):
        if genre in self.genres:
            self.genre = genre
        else:
            raise ValueError(f"Genre '{genre}' not supported.")

    def generate_song(self):
        return [random.choice(self.notes) for _ in range(10)]

    def generate_training_data(self):
        for genre in self.genres:
            self.set_genre(genre)
            self.genre_sequences[genre] = [self.generate_song() for _ in range(10)]
        return self.genre_sequences
