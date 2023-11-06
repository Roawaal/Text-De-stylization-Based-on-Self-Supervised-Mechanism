import torch
import torch.nn as nn
import torch.nn.functional as F
from generator.embedding import EmbeddingLayer
from generator.Encoder import Encoder
from generator.Decoder import Decoder
from generator.outputLinear import OutputLinear


class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, control_embed_size, num_layers, heads, sos_token, eos_token, device):
        super(TextGenerator, self).__init__()
        self.device = device
        self.embed_layer = EmbeddingLayer(vocab_size, embed_size)
        self.encoder = Encoder(control_embed_size, embed_size, num_layers, heads, device)
        self.decoder = Decoder(embed_size, num_layers, heads, device)
        self.output_linear = OutputLinear(embed_size, vocab_size)
        self.control_embedding = nn.Embedding(2, control_embed_size)
        self.SOS_token = sos_token
        self.EOS_token = eos_token

    def forward(self, src, control_code, tgt, src_mask, tgt_mask, batch_size):
        src = self.embed_layer(src)
        tgt = self.embed_layer(tgt)
        control_code_tensor, control_code_mask = self.prepare_control_code(control_code, batch_size)
        control_code_embedding = self.control_embedding(control_code_tensor)
        enc_out = self.encoder(src, control_code_embedding, src_mask, control_code_mask)
        dec_out = self.decoder(tgt, enc_out, control_code_embedding, src_mask, tgt_mask)
        output = self.output_linear(dec_out)
        return output

    def greedy_decode(self, src, control_code, src_mask, max_len, batch_size):
        src = self.embed_layer(src)
        control_code_tensor, control_code_mask = self.prepare_control_code(control_code, batch_size)
        control_code_embedding = self.control_embedding(control_code_tensor)

        enc_out = self.encoder(src, control_code_embedding, src_mask, control_code_mask)

        ys = torch.full((batch_size, 1), fill_value=self.SOS_token).type(torch.long).to(self.device)  # SOS_token is the start token
        for i in range(max_len-1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(self.device)
            out = self.decoder(ys, enc_out, control_code, src_mask, tgt_mask)
            out = self.output_linear(out)
            prob = out[:, -1, :]
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            if next_word == self.EOS_token:
                break
        return ys

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def prepare_control_code(self, control_code, batch_size):
        # Device can be either "cuda" if GPU is available or "cpu"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a tensor for control codes
        control_code_tensor = torch.full((batch_size, 1), fill_value=control_code, dtype=torch.long).to(device)

        # Create a mask for control codes
        control_code_mask = torch.zeros((batch_size, 1, 1, 1), dtype=torch.bool).to(device)

        return control_code_tensor, control_code_mask
