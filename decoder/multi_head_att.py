import torch.nn as nn
from decoder.scaled_dot_prod_att import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0  # Embed size must be divisible by num_heads

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

        # Output linear layer
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_size = x.size()

        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        # Split into multiple heads for parallel attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention for each head
        attention, attention_weights = scaled_dot_product_attention(Q, K, V)

        # Concatenate heads and put through the final linear layer
        attention = (
            attention.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)
        )
        output = self.fc_out(attention)

        return output, attention_weights
