import torch
import numpy as np
import torch.nn as nn

"""
Transformer Decoder Input Layer

Token Embeddings: Convert each note in the sequence into a dense vector representation.
Positional Encodings: Add positional encodings to give the model information about the position of each note in the sequence.
"""


class DecoderInput(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=5000):
        super(DecoderInput, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = self.get_positional_encoding(max_len, embed_size)

    def get_positional_encoding(self, max_len, embed_size):
        pos_encoding = np.zeros((max_len, embed_size))
        positions = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.power(10000, -2 * np.arange(0, embed_size, 2) / embed_size)

        pos_encoding[:, 0::2] = np.sin(positions * div_term)
        pos_encoding[:, 1::2] = np.cos(positions * div_term)

        pos_encoding = torch.FloatTensor(pos_encoding).unsqueeze(0)
        return pos_encoding

    def forward(self, x):
        x = self.token_embedding(x)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x += self.positional_encoding[:, : x.size(1), :].to(device)
        return x
