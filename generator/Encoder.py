import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Encoder(nn.Module):
    def __init__(self, control_embed_size, text_embed_size, num_layers, heads, device):
        super(Encoder, self).__init__()
        self.control_embed_size = control_embed_size
        self.text_embed_size = text_embed_size
        self.device = device

        # control code encoder
        self.control_encoder_layer = TransformerEncoderLayer(control_embed_size, heads)
        self.control_encoder = TransformerEncoder(self.control_encoder_layer, 1)

        # text encoder
        self.text_encoder_layer = TransformerEncoderLayer(text_embed_size, heads)
        self.text_encoder = TransformerEncoder(self.text_encoder_layer, num_layers)

    def forward(self, x, control_code, text_mask, control_code_mask):
        text_encoded = self.text_encoder(x, text_mask)
        control_code_encoded = self.control_code_encoder(control_code, control_code_mask)

        conditional_encoding = text_encoded + control_code_encoded

        return conditional_encoding
