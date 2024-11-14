import torch.nn as nn
from transformer.multi_head_att import MultiHeadAttention
from transformer.feed_forward import FeedForward


class CustomLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size):
        super(CustomLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, src_key_padding_mask=None):
        # Multi-head attention
        attn_output, _ = self.multi_head_attention(x, x, x, mask=src_key_padding_mask)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # Feedforward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x
