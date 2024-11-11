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

# Tokenize the data
tokenized_data = [vocab[token] for token in data]

dummy_input = torch.tensor(tokenized_data).unsqueeze(0).to(device)

dummy_target = torch.tensor(tokenized_data[1:] + [vocab["A"]]).unsqueeze(0).to(device)

index_to_vocab = {v: k for k, v in vocab.items()}

# %%
# Hyperparameters
vocab_size = 10000  # Example vocabulary size
embed_size = 512
num_heads = 8
ff_dim = 2048
num_layers = 6
max_len = 5000
learning_rate = 0.001
num_epochs = 10

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
        print(f"Input sequence: {input_sequence}")
        print(f"Target sequence: {target_sequence}")

        pbar.set_postfix(epoch=epoch + 1, loss=loss.item())
        pbar.update(len(dummy_input))

        wandb.log({"loss": loss.item()})

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

wandb.finish()

# %%

vocab = {"A": 0, "B": 1, "C": 2}
index_to_vocab = {v: k for k, v in vocab.items()}

test_data = ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C", "A"]

tokenized_test_data = [vocab[token] for token in test_data]

test_input = torch.tensor(tokenized_test_data).unsqueeze(0).to(device)

decoder_input.eval()
for layer in decoder_layers:
    layer.eval()
final_layer.eval()

with torch.no_grad():
    x = decoder_input(test_input)
    for layer in decoder_layers:
        x = layer(x)
    output = final_layer(x)

predicted_indices = torch.argmax(output, dim=-1).squeeze(0).tolist()

predicted_tokens = [index_to_vocab[idx] for idx in predicted_indices]

print("Test input sequence:", test_data)
print("Predicted sequence:", predicted_tokens)

# %%
