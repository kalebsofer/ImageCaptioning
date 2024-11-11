import random
import itertools


class NoteGenerator:
    def __init__(self):
        self.notes = ["A", "B", "C", "D", "E", "F", "G"]
        self.genre = "rock"

    def set_genre(self, genre):
        self.genre = genre

    def generate(self):
        while True:
            yield random.choice(self.notes)


if __name__ == "__main__":
    prince = NoteGenerator()
    prince.set_genre("rock")
    stream = prince.generate()

    for _ in range(10):
        print(next(stream))
