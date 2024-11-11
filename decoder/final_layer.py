import torch.nn as nn
import torch.nn.functional as F

"""
Final Layer

Linear Layer (self.fc_out): Projects each embedding of size embed_size to the vocabulary size vocab_size, 
giving a raw score (logit) for each possible next token.

Softmax (F.softmax): Converts the logits into probabilities.
"""


class FinalLayer(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(FinalLayer, self).__init__()
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        logits = self.fc_out(x)
        probs = F.softmax(logits, dim=-1)
        return probs
