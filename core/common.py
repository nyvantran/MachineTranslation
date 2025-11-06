import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Tối ưu hóa bằng cách:
    - Sử dụng F.scaled_dot_product_attention (Flash Attention trong PyTorch 2.0+)
    - Giảm số phép tính lại không cần thiết
    - Giảm số biến trung gian tạm thời
    """

    def __init__(self, emb_dim, num_heads=4, dropout=0.1, at_mask=False):
        """
        Args:
            emb_dim: Kích thước embedding
            num_heads: Số lượng đầu attention
            dropout: Tỷ lệ dropout
            at_mask: Nếu True, sử dụng causal mask (dùng cho decoder)
        """
        super(MultiHeadAttention).__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads if emb_dim % num_heads == 0 else emb_dim // (num_heads - 1)
        self.at_mask = at_mask

        self.QKV_linear = nn.Linear(emb_dim, emb_dim * 3, bias=False)
        self.out_linear = nn.Linear(emb_dim, emb_dim)
        self.dropout = dropout

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, emb_dim) - input tensor
            mask: (attn_mask) (seq_len, seq_len) or (batch_size, seq_len, seq_len) khi evn có PyTorch 2.0+
        Return:
           (batch_size, seq_len, emb_dim)
        """
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
        """
        Args:
            vocab_size: Kích thước từ vựng
            emb_dim: Kích thước embedding
            max_seq_len: Độ dài chuỗi tối đa
            dropout: Tỷ lệ dropout
        """
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


class Feedforward(nn.Module):
    """
    Tối ưu hóa bằng cách:
        - Sử dụng SwiGLU thay vì ReLU
        - Thêm dropout cho regularization
    Ở đây sử dụng SwiGLU: SwiGLU(x) = (Swish(W1(x)) ⊙ W3(x)) W2
       where Swish(x) = x * sigmoid(x) = SiLU(x)
    """

    def __init__(self, d_model=512, d_ff=None, dropout=0.1, bias=False):
        """
        Args:
            d_model: Kích thước embedding đầu vào
            d_ff: Kích thước ẩn của feedforward network
            dropout: Tỷ lệ dropout
            bias: Sử dụng bias trong các lớp Linear hay không
        """
        super(Feedforward, self).__init__()

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
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # SwiGLU(x) = (Swish(W1(x)) ⊙ W3(x)) W2
        # where Swish(x) = x * sigmoid(x) = SiLU(x)
        gate = F.silu(self.w1(x))
        x = self.w3(x)
        x = gate * x  # Element-wise multiplication
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class CrossAttention(nn.Module):
    """
    tối ưu hóa bằng cách:
        - Sử dụng F.scaled_dot_product_attention (Flash Attention trong PyTorch 2.0+)
    """

    def __init__(self, emb_dim, num_heads=8, dropout=0.1, bias=True):
        super(CrossAttention, self).__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dropout_p = dropout

        # Projections
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.value_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias)

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, emb_dim) - from decoder
            key_value: (batch_size, seq_len_kv, emb_dim) - from encoder
            attn_mask: (seq_len_q, seq_len_kv) or (batch_size, seq_len_q, seq_len_kv) khi evn có PyTorch 2.0+
            key_padding_mask: (batch_size, seq_len_kv) - True for positions to ignore khi evn có PyTorch < 2.0
        Returns:
            (batch_size, seq_len_q, emb_dim)
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_kv = key_value.size(1)

        # Project and reshape
        Q = self.query_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = self.key_proj(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        V = self.value_proj(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Use PyTorch's optimized scaled_dot_product_attention
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False  # Cross-attention is not causal
            )
        else:
            # Fallback
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(key_padding_mask, float('-inf'))

            if attn_mask is not None:
                scores = scores + attn_mask

            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)
            out = torch.matmul(attn, V)

        # Reshape and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len_q, self.emb_dim)
        out = self.out_proj(out)

        return out


class Encoder(nn.Module):
    """
    Encoder block gồm:
    - Multi-Head Attention với residual connection và layer normalization
    - Feedforward network với residual connection và layer normalization

    Diagram of Encoder block:

    Encoder
    ├────────────────────────────────┐
    ├───Multi-Head Attention         │
    │   └───Head Attention x 4       │
    ├ + <────────────────────────────┘
    │
    ├───nn.LayerNorm
    ├────────────────────────────────┐
    ├───Feedforward                  │
    ├ + <────────────────────────────┘
    └───nn.LayerNorm
    """

    def __init__(self, dmodel=512, num_heads=4, d_ff=None, dropout=0.1):
        """
        Args:
            dmodel: Kích thước embedding
            num_heads: Số lượng head attention
            dropout: Tỷ lệ dropout
        """
        super(Encoder, self).__init__()
        self.mha = MultiHeadAttention(emb_dim=dmodel, num_heads=num_heads, dropout=dropout)
        self.ffn = Feedforward(d_model=dmodel, dropout=dropout, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, dmodel)
            mask:  (batch_size, seq_len, seq_len) khi evn có PyTorch 2.0+
        Returns:
            (batch_size, seq_len, dmodel)
        """
        # Multi-Head Attention with Residual Connection
        attn_output = self.mha(x, mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feedforward Network with Residual Connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


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
#  └───nn.LayerNorm
class Decoder(nn.Module):
    """
    Decoder block gồm:
    - Masked Multi-Head Attention với residual connection và layer normalization
    - Cross-Attention với residual connection và layer normalization
    - Feedforward network với residual connection và layer normalization
    Diagram of Decoder block:

    Decoder
    ├────────────────────────────────┐
    ├───Masked Multi-Head Attention  │
    │   └───Head Attention x 4       │
    ├ + <────────────────────────────┘
    │
    ├───nn.LayerNorm
    ├────────────────────────────────┐
    ├───Cross-Attention              │
    │   └───Head Attention x 4       │
    ├ + <────────────────────────────┘
    │
    ├───nn.LayerNorm
    ├────────────────────────────────┐
    ├───Feedforward                  │
    ├ + <────────────────────────────┘
    └───nn.LayerNorm
    """

    def __init__(self, dmodel=512, num_heads=4, d_ff=None, dropout=0.1):
        """
        Args:
            dmodel : Kích thước embedding
            num_heads : Số lượng head attention
            dropout : Tỷ lệ dropout
        """
        super(Decoder, self).__init__()
        self.mha = MultiHeadAttention(emb_dim=dmodel, num_heads=num_heads, dropout=dropout, at_mask=True)
        self.cross_attn = CrossAttention(emb_dim=dmodel, num_heads=num_heads, dropout=dropout)
        self.ffn = Feedforward(d_model=dmodel, dropout=dropout, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.norm3 = nn.LayerNorm(dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            X: (batch_size, tgt_seq_len, dmodel) - input tensor to decoder
            enc_output: (batch_size, src_seq_len, dmodel) - output from encoder
            src_mask: (batch_size, tgt_seq_len, src_seq_len) khi evn có PyTorch 2.0+
            tgt_mask: (batch_size, src_seq_len) khi evn có PyTorch < 2.0
        Returns:
            (batch_size, tgt_seq_len, dmodel)
        """
        # Masked Multi-Head Attention with Residual Connection
        attn_output = self.mha(x, mask=src_mask)  # multi-head self-attention
        x = x + self.dropout(attn_output)  # residual connection
        x = self.norm1(x)

        # Cross-Attention with Residual Connection
        cross_attn_output = self.cross_attn(x, enc_output, attn_mask=src_mask,
                                            key_padding_mask=tgt_mask)  # cross-attention
        x = x + self.dropout(cross_attn_output)  # residual connection
        x = self.norm2(x)

        # Feedforward Network with Residual Connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)

        return x
