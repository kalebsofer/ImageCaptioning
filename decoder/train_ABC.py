# %%
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

#
# %%
from decoder.input import DecoderInput
from decoder.decoder_layer import DecoderLayer
from decoder.final_layer import FinalLayer

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
"""
Initinally we will overfit the model to predict the next number in a sequence.
"""
# create some dummy data to overfit the model
data = ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C", "A"]

# Define a simple vocabulary and tokenization
vocab = {"A": 0, "B": 1, "C": 2}  # Example vocabulary

# Create reverse mapping from index to token
index_to_vocab = {index: token for token, index in vocab.items()}

# Tokenize the data
tokenized_data = [vocab[token] for token in data]

dummy_input = torch.tensor(tokenized_data).unsqueeze(0).to(device)

dummy_target = torch.tensor(tokenized_data[1:] + [vocab["A"]]).unsqueeze(0).to(device)

# %%
# Hyperparameters
vocab_size = 3  # Example vocabulary size
embed_size = 16
num_heads = 4
ff_dim = 32
num_layers = 2
max_len = 100
learning_rate = 0.01
num_epochs = 20

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

wandb.init(project="quick one")
# %%
total_steps = num_epochs * len(dummy_input)

with tqdm(total=total_steps, desc="Training Progress") as pbar:
    for epoch in range(num_epochs):
        decoder_input.train()
        for layer in decoder_layers:
            layer.train()
        final_layer.train()

        x = decoder_input(dummy_input)
        for layer in decoder_layers:
            x = layer(x)
        output = final_layer(x)

        loss = criterion(output.view(-1, vocab_size), dummy_target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        input_sequence = [
            index_to_vocab[idx] for idx in dummy_input.squeeze(0).tolist()
        ]
        target_sequence = [
            index_to_vocab[idx] for idx in dummy_target.squeeze(0).tolist()
        ]
        # print(f"Input sequence: {input_sequence}")
        # print(f"Target sequence: {target_sequence}")

        pbar.set_postfix(epoch=epoch + 1, loss=loss.item())
        pbar.update(len(dummy_input))

        wandb.log({"loss": loss.item()})

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

wandb.finish()

# %%

num_predict = 2

# Start with an initial input sequence
initial_sequence = ["A", "A"]
tokenized_initial_sequence = [vocab[token] for token in initial_sequence]
input_sequence = torch.tensor(tokenized_initial_sequence).unsqueeze(0).to(device)

predicted_sequence = initial_sequence.copy()

decoder_input.eval()
for layer in decoder_layers:
    layer.eval()
final_layer.eval()

with torch.no_grad():
    for _ in range(num_predict):
        x = decoder_input(input_sequence)
        for layer in decoder_layers:
            x = layer(x)
        output = final_layer(x)

        # Get the predicted token
        predicted_index = torch.argmax(output, dim=-1).squeeze(0)[-1].item()
        predicted_token = index_to_vocab[predicted_index]

        # Append the predicted token to the sequence
        predicted_sequence.append(predicted_token)

        # Update the input sequence with the new token
        input_sequence = (
            torch.tensor([vocab[token] for token in predicted_sequence])
            .unsqueeze(0)
            .to(device)
        )

print("Generated sequence:", predicted_sequence)

# %%
