# %%
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

# %%
from decoder.input import DecoderInput
from decoder.decoder_layer import DecoderLayer
from decoder.final_layer import FinalLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
import decoder.data.psuedo_music as psuedo_music

note_generator = psuedo_music.NoteGenerator()
training_data = note_generator.generate_training_data()

for genre, songs in training_data.items():
    print(f"Genre: {genre}")
    for song_num, song in enumerate(songs, 1):
        print(f"  Song {song_num}: {song}")

# %%
vocab = {}
index = 0

for genre in training_data.keys():
    genre_token = f"<{genre}>"
    if genre_token not in vocab:
        vocab[genre_token] = index
        index += 1

for genre, songs in training_data.items():
    for song in songs:
        for note in song:
            if note not in vocab:
                vocab[note] = index
                index += 1

index_to_vocab = {index: note for note, index in vocab.items()}

tokenized_data = {}
for genre, songs in training_data.items():
    genre_token = vocab[f"<{genre}>"]
    tokenized_data[genre] = [
        [genre_token] + [vocab[note] for note in song] for song in songs
    ]

inputs = {}
targets = {}
for genre, songs in tokenized_data.items():
    inputs[genre] = [song[:-1] for song in songs]
    targets[genre] = [song[1:] for song in songs]

inputs_tensors = {}
targets_tensors = {}
for genre, songs in inputs.items():
    inputs_tensors[genre] = [torch.tensor(song).to(device) for song in songs]
    targets_tensors[genre] = [torch.tensor(song).to(device) for song in targets[genre]]

# %%
vocab_size = len(vocab)
embed_size = 16
num_heads = 4
ff_dim = 32
num_layers = 2
max_len = 100
learning_rate = 0.001
num_epochs = 300

# %%
decoder_input = DecoderInput(vocab_size, embed_size, max_len).to(device)
decoder_layers = nn.ModuleList(
    [DecoderLayer(embed_size, num_heads, ff_dim).to(device) for _ in range(num_layers)]
)
final_layer = FinalLayer(embed_size, vocab_size).to(device)

# %%
optimizer = optim.Adam(
    list(decoder_input.parameters())
    + list(decoder_layers.parameters())
    + list(final_layer.parameters()),
    lr=learning_rate,
)
criterion = nn.CrossEntropyLoss()

# %%
wandb.init(project="quick music")

total_steps = num_epochs * sum(len(songs) for songs in inputs_tensors.values())

with tqdm(total=total_steps, desc="Training Progress") as pbar:
    for epoch in range(num_epochs):
        decoder_input.train()
        for layer in decoder_layers:
            layer.train()
        final_layer.train()

        for genre in inputs_tensors:
            for input_tensor, target_tensor in zip(
                inputs_tensors[genre], targets_tensors[genre]
            ):
                input_tensor = input_tensor.unsqueeze(0)
                target_tensor = target_tensor.unsqueeze(0)

                x = decoder_input(input_tensor)
                for layer in decoder_layers:
                    x = layer(x)
                output = final_layer(x)

                loss = criterion(output.view(-1, vocab_size), target_tensor.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(epoch=epoch + 1, loss=loss.item())
                pbar.update(1)

                wandb.log({"loss": loss.item()})

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

wandb.finish()

# %%
num_predict = 4

test_cases = [
    ("rock", ["E", "B", "C", "F", "F", "C"]),
    ("jazz", ["E", "A", "B", "B", "F", "E", "F"]),
    ("classical", ["F", "D", "G", "D", "A", "G"]),
    ("pop", ["B", "F", "E", "F", "A", "B"]),
    ("blues", ["D", "E", "A", "C", "C", "F"]),
]

decoder_input.eval()
for layer in decoder_layers:
    layer.eval()
final_layer.eval()

for genre, initial_sequence in test_cases:
    print(f"Testing genre: {genre}, initial sequence: {initial_sequence}")

    genre_token = vocab[f"<{genre}>"]
    tokenized_initial_sequence = [genre_token] + [
        vocab[token] for token in initial_sequence
    ]
    input_sequence = torch.tensor(tokenized_initial_sequence).unsqueeze(0).to(device)

    predicted_sequence = [f"<{genre}>"] + initial_sequence.copy()

    with torch.no_grad():
        for _ in range(num_predict):
            x = decoder_input(input_sequence)
            for layer in decoder_layers:
                x = layer(x)
            output = final_layer(x)

            predicted_index = torch.argmax(output, dim=-1).squeeze(0)[-1].item()
            predicted_token = index_to_vocab[predicted_index]

            predicted_sequence.append(predicted_token)

            input_sequence = (
                torch.tensor([vocab[token] for token in predicted_sequence[1:]])
                .unsqueeze(0)
                .to(device)
            )

    print("Generated sequence:", predicted_sequence[1:])
    print()


# %%
