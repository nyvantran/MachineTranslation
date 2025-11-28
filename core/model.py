import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from core.common import Encoder, Decoder, Embedding


class Transformer(nn.Module):
    """
       Optimized Transformer với:
       - Pre-LayerNorm
       - Optimized components
       - Proper masking
       - Gradient checkpointing support
       - Mixed precision support
       """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_len=5000,
                 pad_idx=(0, 1),
                 use_gradient_checkpointing=False):
        """
        Khởi tạo mô hình Transformer
            Args:
                src_vocab_size (int): Kích thước từ vựng nguồn
                tgt_vocab_size (int): Kích thước từ vựng đích
                d_model (int): Kích thước embedding và mô hình
                num_heads (int): Số lượng đầu attention
                num_encoder_layers (int): Số lớp encoder
                num_decoder_layers (int): Số lớp decoder
                d_ff (int): Kích thước của feed-forward layer
                dropout (float): Tỷ lệ dropout
                max_seq_len (int): Độ dài tối đa của chuỗi
                pad_idx (tuple): Chỉ số padding trong từ vựng
                use_gradient_checkpointing (bool): Sử dụng gradient checkpointing để tiết kiệm bộ nhớ
        """
        super(Transformer, self).__init__()
        # prame init
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # embedding
        self.src_embedding = Embedding(src_vocab_size, d_model, max_seq_len, dropout)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model, max_seq_len, dropout)
        # encoder
        self.encoder = nn.ModuleList([
            Encoder(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        # decoder
        self.decoder = nn.ModuleList([
            Decoder(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        # prediction head
        self.output_layer = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.output_layer.weight = self.tgt_embedding.token_emb.weight
        # reset parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, mask_src=None, mask_tgt=None):
        if mask_src is None:
            mask_src = self.make_src_mask(src)
        if mask_tgt is None:
            mask_tgt = self.make_tgt_mask(tgt)

        encoder_out = self.encode(src, mask_src=mask_src)
        decoder_out = self.decode(tgt=tgt, enc_src=encoder_out, mask_tgt=mask_tgt, mask_src=mask_src)
        out_puts = self.output_layer(decoder_out)
        return out_puts

    def encode(self, src, mask_src=None):
        """Encode source sequence
        Arg:
            src: (batch_size, src_len)
            mask_src: (batch_size, 1, 1, src_len)
        """
        # src: (batch_size, src_len)
        # mask_src: (batch_size, 1, 1, src_len)
        src = self.src_embedding(src)  # (batch_size, src_len, d_model)
        for layer in self.encoder:
            if self.use_gradient_checkpointing:
                src = torch.utils.checkpoint.checkpoint(layer, src, mask_src)  # No tgt for encoder
            else:
                src = layer(src, mask_src)
        src = self.encoder_norm(src)
        return src

    def decode(self, tgt, enc_src, mask_tgt=None, mask_src=None):
        """Decode target sequence
        Arg:
            tgt: (batch_size, tgt_len)
            enc_src: (batch_size, src_len, d_model)
            mask_tgt: (batch_size, 1, tgt_len, tgt_len)
            mask_src: (batch_size, 1, 1, src_len)
        """

        tgt = self.tgt_embedding(tgt)  # (batch_size, tgt_len, d_model)
        for layer in self.decoder:
            if self.use_gradient_checkpointing:
                tgt = torch.utils.checkpoint.checkpoint(
                    layer, tgt, enc_src, mask_tgt, mask_src
                )
            else:
                tgt = layer(tgt, enc_src, mask_tgt, mask_src)
        tgt = self.decoder_norm(tgt)
        return tgt

    def make_src_mask(self, src):
        """Create source padding mask"""
        # src: (batch_size, src_len)
        src_mask = (src == self.pad_idx[0]).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """Create target padding mask"""
        # tgt: (batch_size, tgt_len)
        tgt_mask = (tgt == self.pad_idx[1]).unsqueeze(1).unsqueeze(2)  # (batch_size,tgt_len)

        return tgt_mask

    @torch.no_grad()
    def generate(self, src, max_len=50, start_token=1, end_token=2,
                 temperature=1.0, top_k=None, top_p=None):
        """Generate sequence using greedy decoding or sampling
        Args:
            src: (batch_size, src_len)
            max_len: chiều dài tối đa của chuỗi được tạo
            start_token: index của token bắt đầu
            end_token: index của token kết thúc
            temperature: nhiệt độ cho sampling
            top_k: top-k sampling
            top_p: nucleus sampling
        Returns:
            generated sequences: (batch_size, generated_len)
        """
        self.eval()
        device = src.device

        # Encode source
        if src.dim() == 1:
            src = src.unsqueeze(0)

        src_mask = self.make_src_mask(src)
        enc_src = self.encode(src, src_mask)

        tgt = torch.tensor([[start_token]], device=device)
        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(tgt)
            dec_out = self.decode(tgt, enc_src, mask_src=src_mask, mask_tgt=tgt_mask)  # (batch_size, tgt_len, d_model)
            logits = self.output_layer(dec_out[:, -1, :])  # (batch_size, vocab_size)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Check for end token
            if next_token.item() == end_token:
                break

        return tgt.squeeze(0)  # (generated_len,)

    def init_weights(self):
        """Initialize weights with Xavier uniform"""
        pass


def main():
    import warnings
    import logging
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.utils.checkpoint:*")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Test the Transformer model
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(src_vocab_size, tgt_vocab_size, use_gradient_checkpointing=True).to(device)

    src = torch.randint(0, src_vocab_size, (2, 10)).to(device)  # (batch_size, src_len) (2, 10)
    tgt = torch.randint(0, tgt_vocab_size, (2, 9)).to(device)  # (batch_size, tgt_len) (2, 10)

    out = model(src, tgt)
    print("Output shape:", out.shape)  # Expected: (2, 9, tgt_vocab_size)


if __name__ == "__main__":
    main()
