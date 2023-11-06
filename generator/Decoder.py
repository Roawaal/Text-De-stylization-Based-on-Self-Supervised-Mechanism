import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class Decoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, device):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList([TransformerDecoderLayer(embed_size, heads) for _ in range(num_layers)])
        self.gate = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.Sigmoid()
        )

    def forward(self, x, text_enc_out, control_enc_out, src_mask, tgt_mask):
        out = x
        for layer in self.layers:
            # Apply attention to text encoding and control encoding separately
            text_attention = layer(out, text_enc_out, tgt_mask, src_mask)
            control_attention = layer(out, control_enc_out, tgt_mask, None)

            # Apply gating mechanism
            gate_value = self.gate(torch.cat((text_attention, control_attention), dim=-1))

            # Weighted fusion of text and control attentions
            out = gate_value * text_attention + (1 - gate_value) * control_attention

        return out

