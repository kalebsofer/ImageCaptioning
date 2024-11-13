import torch
import torch.nn.functional as F

"""
Multi-Head Attention Layer

Linear projections: Project the input embeddings into multiple sets of queries (Q), keys (K), and values (V).
Scaled Dot-Product Attention: Compute attention scores between Q and K, scale them, apply masking, and finally, multiply the result by V.
Concatenate and Linear layer: Concatenate the outputs from all attention heads and pass through a final linear layer.
"""


def scaled_dot_product_attention(Q, K, V):
    # dot product of Q and K^T, then scale by sqrt(dimension of key)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )

    # Create lower triangular mask with -inf
    seq_length = scores.size(-1)
    causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, -float("inf"))

    attention_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, V)

    return output, attention_weights
