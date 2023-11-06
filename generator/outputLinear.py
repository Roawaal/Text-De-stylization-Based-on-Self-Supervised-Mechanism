import torch.nn as nn
from torch.nn import functional as F


class OutputLinear(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(OutputLinear, self).__init__()
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=-1)
