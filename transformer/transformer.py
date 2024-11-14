import torch
import torch.nn as nn


# TransformerInput class
class TransformerInput(nn.Module):
    """
    Transformer Input Layer

    Token Embeddings: Convert each token in the sequence into a dense vector representation.
    Rotary Positional Encoding: Add rotary positional encodings to give the model information about the position of each token in the sequence.
    """

    def __init__(self, vocab_size, embed_size, max_len=5000):
        super(TransformerInput, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def positional_encoding(self, x):
        """
        Apply rotary positional encoding to the input tensor.
        """
        seq_len = x.size(1)
        half_dim = self.embed_size // 2

        theta = torch.arange(half_dim, dtype=torch.float32) / half_dim
        theta = 1.0 / (10000**theta)
        theta = theta.unsqueeze(0).unsqueeze(0)

        # Get position IDs
        position_ids = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)
        position_ids = position_ids.unsqueeze(0)

        # Compute sin and cos for each position
        sin = torch.sin(position_ids * theta)
        cos = torch.cos(position_ids * theta)

        # Apply rotary transformation to the input
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return x

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)

        # Rearrange dimensions to match the expected input for TransformerEncoderLayer
        # From (batch_size, seq_len, embed_size) to (seq_len, batch_size, embed_size)
        x = x.permute(1, 0, 2)

        return x


# MultiHeadAttention class
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


# FeedForward class
class FeedForward(nn.Module):
    """
    Feed Forward Layer

    Linear Transformation: Input is projected into a higher-dimensional space.
    ReLU Activation: Adds non-linearity.
    Second Linear Transformation: Projects back to the original dimension, preparing the output for further processing.

    Note: feed forward dimension ff_dim often set to 4*input embedding dimension
    """

    def __init__(self, embed_size, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, embed_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# NormSum class
class NormSum(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super(NormSum, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + self.dropout(sublayer_output))


# CustomLayer class
class CustomLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size):
        super(CustomLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.norm_sum1 = NormSum(embed_size)
        self.norm_sum2 = NormSum(embed_size)

    def forward(self, x, src_key_padding_mask=None):
        # Multi-head attention
        attn_output, _ = self.multi_head_attention(x, x, x, mask=src_key_padding_mask)
        x = self.norm_sum1(x, attn_output)

        # Feedforward network
        ff_output = self.feed_forward(x)
        x = self.norm_sum2(x, ff_output)

        return x


# Main ImageCaptioningTransformer class
class ImageCaptioningTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_hidden_size):
        super(ImageCaptioningTransformer, self).__init__()
        self.input_layer = TransformerInput(vocab_size, embed_size)
        self.encoder_layers = nn.ModuleList(
            [
                CustomLayer(embed_size, num_heads, ff_hidden_size)
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_layer(x)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask)
        x = x.permute(1, 0, 2)  # Rearrange back to (batch_size, seq_len, embed_size)
        x = self.output_layer(x)
        return x
