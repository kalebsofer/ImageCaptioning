import torch
import torch.nn as nn
from transformer.input import TransformerInput
from transformer.layer import CustomLayer

"""
Encoder using BERT-style transformer architecture
"""


class BertEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_hidden_size):
        super(BertEncoder, self).__init__()
        self.input_layer = TransformerInput(vocab_size, embed_size)
        self.layers = nn.ModuleList(
            [
                CustomLayer(embed_size, num_heads, ff_hidden_size)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_key_padding_mask=None):
        # Pass through the input layer to get embedded input with positional encodings
        x = self.input_layer(x)  # Output shape: (seq_len, batch_size, embed_size)

        # Pass through each custom encoder layer
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return x
