import torch.nn as nn

from transformer.multi_head_att import MultiHeadAttention
from transformer.feed_forward import FeedForward

"""
Decoder Layer/Block

Multi-Head Attention with Residual and Normalization: After computing the masked multi-head attention, 
add the result to the original input (x) and normalize it, maintaining the input’s scale and stability.

Feedforward with Residual and Normalization: After passing through the feedforward layer, 
add a residual connection (summing with the input) and apply layer normalization to stabilize the learning process.
"""


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        attention_out, _ = self.attention(x, mask)
        x = self.norm1(x + attention_out)

        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x
