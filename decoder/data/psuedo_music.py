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
        # Generate a sequence of 10 notes for a single song
        return [random.choice(self.notes) for _ in range(10)]

    def generate_training_data(self):
        # Generate 10 songs (each with 10-note sequences) across 5 genres
        for genre in self.genres:
            self.set_genre(genre)
            self.genre_sequences[genre] = [self.generate_song() for _ in range(10)]
        return self.genre_sequences


# Usage example
note_generator = NoteGenerator()
training_data = note_generator.generate_training_data()

# Print the generated training data for each genre
for genre, songs in training_data.items():
    print(f"Genre: {genre}")
    for song_num, song in enumerate(songs, 1):
        print(f"  Song {song_num}: {song}")
