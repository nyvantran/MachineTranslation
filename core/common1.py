import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):  # new
    def __init__(self, emb_dim, num_heads=4, dropout=0.1, at_mask=False):
        super(MultiHeadAttention).__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads if emb_dim % num_heads == 0 else emb_dim // (num_heads - 1)
        self.at_mask = at_mask
        self.dropout = dropout

        self.QKV_linear = nn.Linear(emb_dim, emb_dim * 3, bias=False)
        self.out_linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, emb_dim = x.size()
        qkv = self.QKV_linear(x)  # (batch_size, seq_len, emb_dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (batch_size, num_heads, seq_len, head_dim)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.at_mask
            )  # (batch_size, num_heads, seq_len, head_dim)
        else:
            # Fallback to manual implementation
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            if self.atmask:
                mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)
            attn_output = torch.matmul(attn, v)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, emb_dim)  # (batch_size, seq_len, emb_dim)
        attn_output = self.out_linear(attn_output)  # (batch_size, seq_len, emb_dim)

        return attn_output


class Embedding(nn.Module):
    """
    Tối ưu hóa bằng cách:
    - Cache positional encoding (không tính lại mỗi forward)
    - Sử dụng register_buffer để tự động chuyển device
    - Broadcasting thay vì repeat
    - Pre-compute trong __init__
    - Thêm dropout cho regularization
    """

    def __init__(self, vocab_size, emb_dim=512, max_seq_len=5000, dropout=0.1):
        super(Embedding, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, emb_dim)

        # Pre-compute positional encoding và cache
        pe = self._create_positional_encoding(max_seq_len, emb_dim)
        # Register as buffer (không train, tự động chuyển device)
        self.register_buffer('pe', pe)

    def _create_positional_encoding(self, max_seq_len, emb_dim):
        """Pre-compute positional encoding một lần duy nhất"""
        pe = torch.zeros(max_seq_len, emb_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Tối ưu công thức div_term
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, max_seq_len, emb_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - token indices
        Returns:
            (batch_size, seq_len, emb_dim)
        """
        batch_size, seq_len = x.size()

        # Token embedding
        token_emb = self.token_emb(x)  # (batch_size, seq_len, emb_dim)

        # Positional encoding - sử dụng broadcasting (không cần repeat)
        pos_emb = self.pe[:, :seq_len, :]  # (1, seq_len, emb_dim)

        # Combine (broadcasting tự động)
        x = token_emb + pos_emb

        return self.dropout(x)


class SwiGLUFeedforward(nn.Module):
    """
    SwiGLU như trong LLaMA - state-of-the-art
    Paper: "GLU Variants Improve Transformer"
    """

    def __init__(self, d_model=512, d_ff=None, dropout=0.1, bias=False):
        super(SwiGLUFeedforward, self).__init__()

        if d_ff is None:
            # LLaMA uses 8/3 * d_model to compensate for gating
            d_ff = int(8 * d_model / 3)
            # Round to nearest multiple of 256 for efficiency
            d_ff = 256 * ((d_ff + 255) // 256)

        # Three projections: W1 (gate), W2 (down), W3 (up)
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # Gate
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)  # Up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU(x) = (Swish(W1(x)) ⊙ W3(x)) W2
        # where Swish(x) = x * sigmoid(x) = SiLU(x)
        gate = F.silu(self.w1(x))
        x = self.w3(x)
        x = gate * x  # Element-wise multiplication
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x
