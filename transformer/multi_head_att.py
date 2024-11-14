import torch
import torch.nn as nn
from decoder.scaled_dot_prod_att import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size, num_heads, kqv_dim=None):
        """
        Multi-head attention module.

        Args:
            embed_size (int): The embedding size for queries.
            num_heads (int): The number of attention heads.
            kv_embed_size (int, optional): The embedding size for keys and values. Defaults to embed_size.
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embed size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # If kqv_dim is not provided, assume it's the same as embed_size
        self.kqv_dim = kqv_dim if kqv_dim is not None else embed_size

        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(self.kqv_dim, embed_size)
        self.value_linear = nn.Linear(self.kqv_dim, embed_size)

        # Output linear layer
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.query_linear(Q)
        K = self.key_linear(K)
        V = self.value_linear(V)

        # Split into multiple heads for parallel attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention for each head
        attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and put through the final linear layer
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        output = self.fc_out(attention)

        return output, attention_weights
