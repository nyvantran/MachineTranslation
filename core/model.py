import torch
import torch.nn as nn
from core.common import Embedding, Encoder, Decoder, PredictionHead

architecture = {
    "transformer": {
        'embedding': {'input_dim': 100000, 'emb_dim': 512},
        'encoder': {'dmodel': 512, 'num_heads': 4, 'num_layers': 6},
        'decoder': {'dmodel': 512, 'num_heads': 4, 'num_layers': 6},
        'output_layer': {'output_dim': 100000}
    }
}


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim=512, num_heads=4, num_layers=6):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding_input = Embedding(input_dim=input_dim, emb_dim=emb_dim)
        self.embedding_output = Embedding(input_dim=output_dim, emb_dim=emb_dim)
        self.module_layers = nn.ModuleDict()
        self.setup_module()

    def setup_module(self):
        # setup encoder layers
        encoder_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            encoder_layers.append(Encoder(dmodel=512, num_heads=self.num_heads))
        self.module_layers['encoder'] = encoder_layers
        # setup decoder layers
        decoder_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            decoder_layers.append(Decoder(dmodel=512, num_heads=self.num_heads))
        self.module_layers['decoder'] = decoder_layers
        # setup prediction head
        self.module_layers['prediction_head'] = PredictionHead(dmodel=512, output_dim=self.output_dim)

    def forward(self, src, tgt):
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        src_emb = self.embedding_input(src)  # (batch_size, src_seq_len, emb_dim)
        tgt_emb = self.embedding_output(tgt)  # (batch_size, tgt_seq_len, emb_dim)

        # Encoder
        encoder_output = src_emb
        for layer in self.module_layers['encoder']:
            encoder_output = layer(encoder_output)

        # Decoder
        decoder_output = tgt_emb
        for layer in self.module_layers['decoder']:
            decoder_output = layer(decoder_output, encoder_output)

        # Prediction Head
        output = self.module_layers['prediction_head'](decoder_output)  # (batch_size, tgt_seq_len, output_dim)
        return output


def main():
    x = torch.randint(0, 100000, (32, 10))  # (batch_size, seq_len)
    y = torch.randint(0, 100000, (32, 12))  # (batch_size, seq_len)
    model = Transformer(input_dim=100000, output_dim=100000, emb_dim=512, num_heads=4, num_layers=6)
    output = model(x, y)
    print("output", output.shape)  # expected output: (32, 12, 100000)


if __name__ == "__main__":
    main()
