import torch.nn as nn
from generator.PositionalEncoding import PositionalEncoding


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, ):
        super(EmbeddingLayer, self).__init__()
        # Word embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        # Position embedding
        self.position_embedding = PositionalEncoding(embed_size)

    def forward(self, x):
        # Get word embedding
        token_embed = self.token_embedding(x)
        # Get position embedding
        position_embed = self.position_embedding(x)

        return token_embed + position_embed

