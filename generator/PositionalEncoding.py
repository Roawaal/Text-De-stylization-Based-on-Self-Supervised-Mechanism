import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: Dimension of the model (usually the same as embedding dimension)
        :param max_len: Maximum length of a sentence for which positional encoding will be pre-computed
        """
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: Input data
        :return: Positional encoding corresponding to input data
        """
        return self.pe[:, :x.size(1)]
