import torch.nn as nn

"""
Feed Forward Layer

Linear Transformation: Input is projected into a higher-dimensional space.
ReLU Activation: Adds non-linearity.
Second Linear Transformation: Projects back to the original dimension, preparing the output for further processing.

Note: feed forward dimension ff_dim often set to 4*input embedding dimension
"""


class FeedForward(nn.Module):
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
