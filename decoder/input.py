import torch
import torch.nn as nn

"""
Transformer Decoder Input Layer

Token Embeddings: Convert each note in the sequence into a dense vector representation.
Rotary Positional Encoding: Add rotary positional encodings to give the model information about the position of each note in the sequence.
"""


class DecoderInput(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=5000):
        super(DecoderInput, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def apply_rotary_positional_encoding(self, x):
        seq_len = x.size(1)
        half_dim = self.embed_size // 2
        theta = torch.arange(half_dim, dtype=torch.float32) / half_dim
        theta = 1.0 / (10000**theta)
        theta = theta.unsqueeze(0).unsqueeze(0)

        position_ids = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)
        position_ids = position_ids.unsqueeze(0)

        sin = torch.sin(position_ids * theta)
        cos = torch.cos(position_ids * theta)

        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.apply_rotary_positional_encoding(x)
        return x
