import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import random
import sentencepiece as spm
import math
import pandas as pd


class Attention(nn.Module):
    def __init__(self, d, num_heads, is_causal=False):
        super().__init__()
        self.is_causal = is_causal
        self.num_heads = num_heads
        self.d_per_head = d // num_heads
        self.scaling = self.d_per_head**-0.5

        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        self.fc = nn.Linear(d, d)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_length, d = q.size()
        # Ensure the input size matches the expected size
        assert d == self.num_heads * self.d_per_head, "Input dimension mismatch"

        Q = (
            self.W_Q(q)
            .reshape(batch_size, seq_length, self.num_heads, self.d_per_head)
            .transpose(1, 2)
            .contiguous()
        )
        K = (
            self.W_K(k)
            .reshape(batch_size, -1, self.num_heads, self.d_per_head)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            self.W_V(v)
            .reshape(batch_size, -1, self.num_heads, self.d_per_head)
            .transpose(1, 2)
            .contiguous()
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, d)

        return self.fc(output)


class FeedForward(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, d * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d * 4, d)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.attention = Attention(d, num_heads)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = FeedForward(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class Encoder(nn.Module):
    def __init__(self, d, num_heads, num_layers):
        super().__init__()
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(
            *(list(self.cnn.children())[:-2])
        )  # Keep spatial dimensions
        self.linear = nn.Linear(2048, d)
        self.layers = nn.ModuleList(
            [EncoderBlock(d, num_heads) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)
        cnn_features = self.cnn(x)  # Shape: [batch_size, 2048, 7, 7]

        # Adjust the reshaping to match the actual spatial dimensions
        cnn_features = cnn_features.view(batch_size, 2048, -1).transpose(
            1, 2
        )  # Shape: [batch_size, 49, 2048] if 7x7, or [batch_size, 64, 2048] if 8x8

        x = self.linear(cnn_features)  # Shape: [batch_size, seq_length, d]
        seq_length = x.size(1)

        # Generate positional encoding for the actual sequence length
        pos_encoding = self.rotary_positional_encoding(seq_length, x.size(2))
        x = x + pos_encoding

        for layer in self.layers:
            x = layer(x)

        return x

    @staticmethod
    def rotary_positional_encoding(seq_length, d_model):
        position = torch.arange(0, seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class DecoderBlock(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.self_attn = Attention(d, num_heads, is_causal=True)
        self.cross_attn = Attention(d, num_heads)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = FeedForward(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, x, encoder_output, mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask))
        batch_size = x.size(0)
        encoder_batch_size = encoder_output.size(0)
        if batch_size != encoder_batch_size:
            encoder_output = encoder_output.repeat(
                batch_size // encoder_batch_size, 1, 1
            )
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, mask)
        x = self.norm2(x + cross_attn_output)
        x = self.norm3(x + self.ffn(x))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d)
        self.pos_encoding = nn.Parameter(
            self.rotary_positional_encoding(100, d), requires_grad=False
        )
        self.layers = nn.ModuleList(
            [DecoderBlock(d, num_heads) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, encoder_output, mask=None):
        x = self.embedding(x) + self.pos_encoding[: x.size(1), :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, mask)

        return self.fc(x)

    @staticmethod
    def rotary_positional_encoding(seq_length, d_model):
        position = torch.arange(0, seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerB(nn.Module):

    def __init__(
        self,
        vocab_size,
        d=512,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=8,
    ):
        super().__init__()
        self.encoder = Encoder(d, num_heads, num_encoder_layers)
        self.decoder = Decoder(vocab_size, d, num_heads, num_decoder_layers)

    def forward(self, image, caption, mask=None):
        encoder_output = self.encoder(image)
        decoder_output = self.decoder(caption, encoder_output, mask)
        return decoder_output


if __name__ == "__main__":
    vocab_size = 16000
    model = TransformerB(vocab_size=vocab_size)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Parameters: ", params)
    print(model)
