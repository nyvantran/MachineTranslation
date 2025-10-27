import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, input_dim, emb_dim=512):
        super(Embedding, self).__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)

    def token_embedding(self, x):
        #  x:(batch_size, seq_len, emb_dim)
        #  output emd shape:
        #       +─── emb_dim = 512 ──>
        #       │
        #    seq_len
        #       │
        #       v
        return self.embedding(x)  # (batch_size, seq_len, emb_dim)

    def position_embedding(self, x):
        batch_size, seq_len, emb_dim = x.size()
        pe = torch.zeros(seq_len, emb_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
        return pe

    def forward(self, x):
        emd = self.token_embedding(x)
        pos = self.position_embedding(emd).to(x.device)
        return torch.add(emd, pos)


#  define module structures of Encoder block
#  diagram of Encoder block
#  Encoder
#  ├────────────────────────────────┐
#  ├───Multi-Head Attention         │
#  │   └───Head Attention x 4       │
#  ├ + <────────────────────────────┘
#  │
#  ├───nn.LayerNorm
#  ├────────────────────────────────┐
#  ├───Feedforward                  │
#  ├ + <────────────────────────────┘
#  └───nn.LayerNorm


class Encoder(nn.Module):
    def __init__(self, dmodel=512, num_heads=4):
        super(Encoder, self).__init__()
        self.num_heads = num_heads
        self.dmodel = dmodel
        #
        self.MHAttention = MultiHeadAttention(emb_dim=dmodel, num_heads=num_heads)
        self.LayerNorm1 = nn.LayerNorm(dmodel)
        self.Feedforward = Feedforward(dmodel=dmodel)
        self.LayerNorm2 = nn.LayerNorm(dmodel)

    def forward(self, x):
        # x: (batch_size, seq_len, dmodel)
        lnorm1 = self.LayerNorm1(self.MHAttention(x) + x)  # (batch_size, seq_len, dmodel)
        lnorm2 = self.LayerNorm2(self.Feedforward(lnorm1) + lnorm1)  # (batch_size, seq_len, dmodel)
        return lnorm2


class Feedforward(nn.Module):
    def __init__(self, dmodel=512):
        super(Feedforward, self).__init__()
        self.linear1 = nn.Linear(dmodel, dmodel * 2, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dmodel * 2, dmodel, bias=True)

    def forward(self, x):
        linear1 = self.linear1(x)
        relu = self.relu(linear1)
        linear2 = self.linear2(relu)
        return linear2


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=4, atmask=False):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.atmask = atmask

        self.head_dim = emb_dim // num_heads if emb_dim % num_heads == 0 else emb_dim // (num_heads - 1)
        self.heads = nn.ModuleList(
            [HeadAttention(self.emb_dim, self.head_dim, atmask=self.atmask) for _ in range(num_heads)])
        self.combo_linear = nn.Linear(self.head_dim * num_heads, emb_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, emb_dim)
        head_outputs = [head(x) for head in self.heads]
        # head_outputs: list of (batch_size, seq_len, head_dim)
        concat = torch.cat(head_outputs, dim=-1)
        output = self.combo_linear(concat)
        return output


class HeadAttention(nn.Module):
    def __init__(self, emb_dim=512, head_dim=512, atmask=False):
        super(HeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.atmask = atmask

        self.query_linear = nn.Linear(emb_dim, head_dim)
        self.key_linear = nn.Linear(emb_dim, head_dim)
        self.value_linear = nn.Linear(emb_dim, head_dim)
        self.scale = torch.sqrt(torch.FloatTensor([head_dim]))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch_size, seq_len, emb_dim)
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        # query, key, value: (batch_size, seq_len, head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale.to(query.device)
        if self.atmask:
            # create mask
            batch_size, seq_len, _ = x.size()
            mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask.unsqueeze(0).expand(batch_size, -1, -1), float('-inf'))
        attn = self.softmax(scores)  # (batch_size, seg_query_len, seq_key_len)
        output = torch.matmul(attn, value)
        return output


# define module structures of Decoder block
#  diagram of Decoder block
#  Decoder
#  ├────────────────────────────────┐
#  ├───Masked Multi-Head Attention  │
#  │   └───Head Attention x 4       │
#  ├ + <────────────────────────────┘
#  │
#  ├───nn.LayerNorm
#  ├────────────────────────────────┐
#  ├───Cross-Attention              │
#  │   └───Head Attention x 4       │
#  ├ + <────────────────────────────┘
#  │
#  ├───nn.LayerNorm
#  ├────────────────────────────────┐
#  ├───Feedforward                  │
#  ├ + <────────────────────────────┘
class Decoder(nn.Module):
    def __init__(self, dmodel=512, num_heads=4):
        super(Decoder, self).__init__()
        self.num_heads = num_heads
        self.dmodel = dmodel
        #
        self.MHAttention = MultiHeadAttention(emb_dim=dmodel, num_heads=num_heads, atmask=True)
        self.LayerNorm1 = nn.LayerNorm(dmodel)
        self.CrossAttention = CrossMultiHeadAttention(emb_dim=dmodel, num_heads=num_heads)
        self.LayerNorm2 = nn.LayerNorm(dmodel)
        self.Feedforward = Feedforward(dmodel=dmodel)
        self.LayerNorm3 = nn.LayerNorm(dmodel)

    def forward(self, x, enc_output):
        # x: (batch_size, seq_len, dmodel)
        lnorm1 = self.LayerNorm1(self.MHAttention(x) + x)  # (batch_size, seq_len, dmodel)
        lnorm2 = self.LayerNorm2(self.CrossAttention(lnorm1, enc_output) + lnorm1)  # (batch_size, seq_len, dmodel)
        lnorm3 = self.LayerNorm3(self.Feedforward(lnorm2) + lnorm2)  # (batch_size, seq_len, dmodel)
        return lnorm3


class crossAttention(nn.Module):
    def __init__(self, emb_dim=512, head_dim=512):
        super(crossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.head_dim = head_dim

        self.query_linear = nn.Linear(emb_dim, head_dim)
        self.key_linear = nn.Linear(emb_dim, head_dim)
        self.value_linear = nn.Linear(emb_dim, head_dim)
        self.scale = torch.sqrt(torch.FloatTensor([head_dim]))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_in, encoder_out):
        # decoder_in: (batch_size, seq_len_query, emb_dim)
        # encoder_out: (batch_size, seq_len_key, emb_dim)
        keys = self.key_linear(encoder_out)
        query = self.query_linear(decoder_in)
        value = self.value_linear(encoder_out)
        # query, key, value: (batch_size, seq_len, head_dim)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale.to(keys.device)
        attn = self.softmax(scores)  # (batch_size, seg_query_len, seq_key_len)
        output = torch.matmul(attn, value)  # (batch_size, seq_len_query, head_dim)
        return output


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=4):
        super(CrossMultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        self.head_dim = emb_dim // num_heads if emb_dim % num_heads == 0 else emb_dim // (num_heads - 1)
        self.heads = nn.ModuleList(
            [crossAttention(self.emb_dim, self.head_dim) for _ in range(num_heads)])
        self.combo_linear = nn.Linear(self.head_dim * num_heads, emb_dim)

    def forward(self, x, y):
        # x: query (batch_size, seq_len_query, emb_dim)
        # y: key, value (batch_size, seq_len_key, emb_dim)
        head_outputs = [head(x, y) for head in self.heads]
        # head_outputs: list of (batch_size, seq_len_query, head_dim)
        concat = torch.cat(head_outputs, dim=-1)
        output = self.combo_linear(concat)
        return output


class PredictionHead(nn.Module):
    def __init__(self, dmodel=512, output_dim=1000):
        super(PredictionHead, self).__init__()
        self.linear = nn.Linear(dmodel, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, dmodel)
        output = self.linear(x)  # (batch_size, seq_len, output_dim)
        return output


# │   ├─── └───
def main():
    x = torch.ones((1, 5)).int()  # (batch_size, seq_len)
    embedding = Embedding(input_dim=1000, emb_dim=512)
    emd_output = embedding(x)  # (batch_size, seq_len, emb_dim)
    # print("Embedding output shape:", emd_output.shape)


if __name__ == "__main__":
    main()
