import torch
import torch.nn as nn
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
                 pad_idx=0,
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
                pad_idx (int): Chỉ số padding trong từ vựng
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
        # src: (batch_size, src_len)
        pass

    def encode(self, src, mask_src=None):
        """Encode source sequence"""
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
        """Decode target sequence"""
        # tgt: (batch_size, tgt_len)
        # enc_src: (batch_size, src_len, d_model)
        # mask_tgt: (batch_size, 1, tgt_len, tgt_len)
        # mask_src: (batch_size, 1, 1, src_len)
        tgt = self.tgt_embedding(tgt)  # (batch_size, tgt_len, d_model)
        for layer in self.decoder:
            if self.use_gradient_checkpointing:
                tgt = torch.utils.checkpoint.checkpoint(layer, tgt, enc_src, mask_tgt, mask_src)
            else:
                tgt = layer(tgt, enc_src, mask_tgt, mask_src)
        tgt = self.decoder_norm(tgt)
        output = self.output_layer(tgt)  # (batch_size, tgt_len, tgt_vocab_size)
        return output

    def make_src_mask(self, src):
        """Create source padding mask"""
        # src: (batch_size, src_len)
        src_mask = (src == self.pad_idx)
        return src_mask

    def make_tgt_mask(self, tgt):
        """Create target padding mask"""
        # tgt: (batch_size, tgt_len)
        tgt_mask = (tgt == self.pad_idx)
        return tgt_mask
