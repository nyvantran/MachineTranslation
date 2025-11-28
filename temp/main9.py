import torch
import torch.nn as nn
import math


def init_tfixup_transformer(model: nn.Transformer):
    """
    Áp dụng khởi tạo T-Fixup cho một mô hình nn.Transformer của Pytorch.
    """

    # Lấy số lượng lớp L
    # Chúng ta sẽ giả sử số lớp encoder và decoder là như nhau
    # Nếu chúng khác nhau, bạn cần xử lý riêng L_encoder và L_decoder
    num_encoder_layers = model.encoder.num_layers
    num_decoder_layers = model.decoder.num_layers

    # Lấy d_model từ lớp embedding (giả sử nó tồn tại và là
    # `model.encoder.layers[0].self_attn.embed_dim` hoặc tương tự)
    # Tuy nhiên, cách an toàn hơn là lấy từ một lớp linear
    try:
        d_model = model.encoder.layers[0].linear1.in_features
    except AttributeError:
        print("Không thể tự động phát hiện d_model. Đặt mặc định là 512.")
        d_model = 512

    print(f"Áp dụng T-Fixup với L_enc={num_encoder_layers}, L_dec={num_decoder_layers}, d_model={d_model}")

    # 1. Khởi tạo Embeddings
    # Giả sử mô hình có các lớp embedding tên là 'src_embed' và 'tgt_embed'
    # Nếu bạn dùng nn.Transformer, bạn phải tự thêm các lớp embedding này
    # Ở đây, chúng ta sẽ khởi tạo các lớp embedding của chính nn.Transformer
    # (nếu chúng được định nghĩa bên ngoài)
    #
    # Ví dụ nếu bạn có: self.src_tok_emb = nn.Embedding(...)
    # self.pos_emb = nn.Embedding(...)

    # Giả sử chúng ta khởi tạo các lớp embedding bên ngoài mô hình
    # và truyền chúng vào. Đoạn code này tập trung vào các lớp
    # bên trong nn.Transformer.

    # 2. Khởi tạo các lớp Encoder
    L_enc = num_encoder_layers
    scale_enc = L_enc ** -0.5  # L^(-0.5)

    for layer in model.encoder.layers:
        # --- Multi-Head Attention (MHA) ---

        # Lớp "Input/Value" (Q, K, V projection)
        # nn.Transformer gộp Q, K, V vào in_proj
        if hasattr(layer.self_attn, 'in_proj_weight'):
            nn.init.xavier_normal_(layer.self_attn.in_proj_weight, gain=1.0)
            layer.self_attn.in_proj_weight.data *= scale_enc
            nn.init.zeros_(layer.self_attn.in_proj_bias)

        # Lớp "Output" (Output projection)
        if hasattr(layer.self_attn, 'out_proj'):
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

        # --- Feed-Forward Network (FFN) ---

        # Lớp "Input/Value" (Linear 1)
        nn.init.xavier_normal_(layer.linear1.weight, gain=1.0)
        layer.linear1.weight.data *= scale_enc
        nn.init.zeros_(layer.linear1.bias)

        # Lớp "Output" (Linear 2)
        nn.init.zeros_(layer.linear2.weight)
        nn.init.zeros_(layer.linear2.bias)

        # --- LayerNorm (nếu giữ lại) ---
        # T-Fixup "thuần" sẽ loại bỏ chúng.
        # Nếu giữ lại, chúng ta khởi tạo chúng về 1 và 0.
        nn.init.constant_(layer.norm1.weight, 1.0)
        nn.init.zeros_(layer.norm1.bias)
        nn.init.constant_(layer.norm2.weight, 1.0)
        nn.init.zeros_(layer.norm2.bias)

    # 3. Khởi tạo các lớp Decoder
    # Tương tự như Encoder
    L_dec = num_decoder_layers
    scale_dec = L_dec ** -0.5  # L^(-0.5)

    for layer in model.decoder.layers:
        # --- Self-Attention (MHA) ---
        if hasattr(layer.self_attn, 'in_proj_weight'):
            nn.init.xavier_normal_(layer.self_attn.in_proj_weight, gain=1.0)
            layer.self_attn.in_proj_weight.data *= scale_dec
            nn.init.zeros_(layer.self_attn.in_proj_bias)

        if hasattr(layer.self_attn, 'out_proj'):
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

        # --- Multi-Head Cross-Attention ---
        if hasattr(layer.multihead_attn, 'in_proj_weight'):
            nn.init.xavier_normal_(layer.multihead_attn.in_proj_weight, gain=1.0)
            layer.multihead_attn.in_proj_weight.data *= scale_dec
            nn.init.zeros_(layer.multihead_attn.in_proj_bias)

        if hasattr(layer.multihead_attn, 'out_proj'):
            nn.init.zeros_(layer.multihead_attn.out_proj.weight)
            nn.init.zeros_(layer.multihead_attn.out_proj.bias)

        # --- Feed-Forward Network (FFN) ---
        nn.init.xavier_normal_(layer.linear1.weight, gain=1.0)
        layer.linear1.weight.data *= scale_dec
        nn.init.zeros_(layer.linear1.bias)

        nn.init.zeros_(layer.linear2.weight)
        nn.init.zeros_(layer.linear2.bias)

        # --- LayerNorm (nếu giữ lại) ---
        nn.init.constant_(layer.norm1.weight, 1.0)
        nn.init.zeros_(layer.norm1.bias)
        nn.init.constant_(layer.norm2.weight, 1.0)
        nn.init.zeros_(layer.norm2.bias)
        nn.init.constant_(layer.norm3.weight, 1.0)
        nn.init.zeros_(layer.norm3.bias)

    # 4. Khởi tạo các lớp Norm cuối (nếu có)
    if model.encoder.norm:
        nn.init.constant_(model.encoder.norm.weight, 1.0)
        nn.init.zeros_(model.encoder.norm.bias)
    if model.decoder.norm:
        nn.init.constant_(model.decoder.norm.weight, 1.0)
        nn.init.zeros_(model.decoder.norm.bias)

    print("Hoàn tất khởi tạo T-Fixup.")


# --- Cách sử dụng ---

# 1. Định nghĩa mô hình
d_model = 512
nhead = 8
num_encoder_layers = 12  # L = 12
num_decoder_layers = 12  # L = 12
dim_feedforward = 2048
dropout = 0.1

transformer_model = nn.Transformer(
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=True  # Sử dụng batch_first=True cho dễ
)

# 2. Áp dụng hàm khởi tạo T-Fixup
init_tfixup_transformer(transformer_model)

# 3. Kiểm tra một vài trọng số
print("\n--- Kiểm tra trọng số (Encoder Layer 0) ---")
print("MHA in_proj (đã scale):", transformer_model.encoder.layers[0].self_attn.in_proj_weight.data.std())
print("MHA out_proj (về 0):", transformer_model.encoder.layers[0].self_attn.out_proj.weight.data.std())
print("FFN linear1 (đã scale):", transformer_model.encoder.layers[0].linear1.weight.data.std())
print("FFN linear2 (về 0):", transformer_model.encoder.layers[0].linear2.weight.data.std())
