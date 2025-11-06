import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


class OptimizedCrossMultiHeadAttention(nn.Module):
    """
    Tối ưu hóa bằng cách:
    - Batch tất cả heads vào 1 phép toán
    - Cache scale factor
    - Sử dụng tensor operations thay vì list
    - Thêm dropout, bias
    - Efficient memory usage
    """

    def __init__(self, emb_dim, num_heads=8, dropout=0.1, bias=True):
        super(OptimizedCrossMultiHeadAttention, self).__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Projections for Q, K, V
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.value_proj = nn.Linear(emb_dim, emb_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias)

        # Register scale as buffer
        self.register_buffer('scale', torch.tensor(1.0 / math.sqrt(self.head_dim)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, emb_dim) - from decoder
            key_value: (batch_size, seq_len_kv, emb_dim) - from encoder
            attn_mask: (seq_len_q, seq_len_kv) or (batch_size, seq_len_q, seq_len_kv)
            key_padding_mask: (batch_size, seq_len_kv) - True for positions to ignore
        Returns:
            (batch_size, seq_len_q, emb_dim)
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_kv = key_value.size(1)

        # Project and reshape to (batch_size, num_heads, seq_len, head_dim)
        Q = self.query_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = self.key_proj(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        V = self.value_proj(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: (batch_size, num_heads, seq_len_q, seq_len_kv)

        # Apply masks
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, seq_len_kv)
            # Expand to (batch_size, 1, 1, seq_len_kv)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))

        if attn_mask is not None:
            scores = scores + attn_mask

        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (batch_size, num_heads, seq_len_q, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous()  # (batch_size, seq_len_q, num_heads, head_dim)
        out = out.view(batch_size, seq_len_q, self.emb_dim)  # (batch_size, seq_len_q, emb_dim)

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class FusedCrossAttention(nn.Module):
    """
    Sử dụng F.scaled_dot_product_attention (Flash Attention trong PyTorch 2.0+)
    """

    def __init__(self, emb_dim, num_heads=8, dropout=0.1, bias=True):
        super(FusedCrossAttention, self).__init__()
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


class EfficientCrossAttention(nn.Module):
    """
    Memory-efficient cross-attention với các tối ưu bổ sung
    """

    def __init__(self, emb_dim, num_heads=8, dropout=0.1, bias=True,
                 use_kv_cache=False):
        super(EfficientCrossAttention, self).__init__()
        assert emb_dim % num_heads == 0

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dropout_p = dropout
        self.use_kv_cache = use_kv_cache

        # Single projection for query
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=bias)

        # Combined projection for key and value
        self.kv_proj = nn.Linear(emb_dim, 2 * emb_dim, bias=bias)

        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias)

        # KV cache for inference
        self.register_buffer('cached_k', None)
        self.register_buffer('cached_v', None)

    def forward(self, query, key_value, use_cache=False):
        batch_size, seq_len_q, _ = query.size()

        # Project query
        Q = self.q_proj(query)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)

        # Project key and value together
        if use_cache and self.cached_k is not None:
            K, V = self.cached_k, self.cached_v
        else:
            KV = self.kv_proj(key_value)
            KV = KV.view(batch_size, -1, 2, self.num_heads, self.head_dim)
            K, V = KV[:, :, 0], KV[:, :, 1]
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            if use_cache:
                self.cached_k = K
                self.cached_v = V

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.emb_dim)
        out = self.out_proj(out)

        return out

    def clear_cache(self):
        self.cached_k = None
        self.cached_v = None


class LinearAttention(nn.Module):
    """
    Linear Attention - O(N) complexity instead of O(N²)
    Useful for very long sequences
    """

    def __init__(self, emb_dim, num_heads=8, dropout=0.1):
        super(LinearAttention, self).__init__()
        assert emb_dim % num_heads == 0

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        batch_size, seq_len_q, _ = query.size()
        seq_len_kv = key_value.size(1)

        # Project
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = self.k_proj(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        V = self.v_proj(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim)

        # Apply ELU + 1 for positive features (kernel trick)
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # Transpose for computation
        Q = Q.transpose(1, 2)  # (B, H, T_q, D)
        K = K.transpose(1, 2)  # (B, H, T_kv, D)
        V = V.transpose(1, 2)  # (B, H, T_kv, D)

        # Linear attention: O(N) instead of O(N²)
        # Compute K^T V first: (B, H, D, D)
        KV = torch.matmul(K.transpose(-2, -1), V)

        # Normalization term
        K_sum = K.sum(dim=-2, keepdim=True)  # (B, H, 1, D)

        # Compute attention output
        # Q @ (K^T @ V) / (Q @ K_sum)
        out = torch.matmul(Q, KV)  # (B, H, T_q, D)
        normalizer = torch.matmul(Q, K_sum.transpose(-2, -1))  # (B, H, T_q, 1)
        out = out / (normalizer + 1e-6)

        # Reshape
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len_q, self.emb_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


# Original implementations (cleaned)
class OriginalCrossAttention(nn.Module):
    def __init__(self, emb_dim=512, head_dim=512):
        super(OriginalCrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.head_dim = head_dim

        self.query_linear = nn.Linear(emb_dim, head_dim, bias=False)
        self.key_linear = nn.Linear(emb_dim, head_dim, bias=False)
        self.value_linear = nn.Linear(emb_dim, head_dim, bias=False)
        self.scale = torch.sqrt(torch.FloatTensor([head_dim]))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_in, encoder_out):
        keys = self.key_linear(encoder_out)
        query = self.query_linear(decoder_in)
        value = self.value_linear(encoder_out)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale.to(keys.device)
        attn = self.softmax(scores)
        output = torch.matmul(attn, value)
        return output


class OriginalCrossMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=4):
        super(OriginalCrossMultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.heads = nn.ModuleList(
            [OriginalCrossAttention(self.emb_dim, self.head_dim) for _ in range(num_heads)])
        self.combo_linear = nn.Linear(self.head_dim * num_heads, emb_dim, bias=False)

    def forward(self, x, y):
        head_outputs = [head(x, y) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        output = self.combo_linear(concat)
        return output


class PyTorchNativeCrossAttention(nn.Module):
    """Wrapper cho nn.MultiheadAttention của PyTorch"""

    def __init__(self, emb_dim, num_heads=8, dropout=0.1, bias=True):
        super(PyTorchNativeCrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            emb_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True
        )

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, emb_dim)
            key_value: (batch_size, seq_len_kv, emb_dim)
        """
        out, _ = self.mha(
            query, key_value, key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        return out


def count_parameters(model):
    """Đếm số parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_cross_attention(model, batch_size, seq_len_q, seq_len_kv, emb_dim,
                              num_iterations=1000, warmup=100):
    """Benchmark cross-attention layers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    query = torch.randn(batch_size, seq_len_q, emb_dim, device=device)
    key_value = torch.randn(batch_size, seq_len_kv, emb_dim, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(query, key_value)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(query, key_value)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(query, key_value)
        elapsed_time = (time.time() - start_time) * 1000
        avg_time = elapsed_time / num_iterations
        peak_memory = 0

    return avg_time, peak_memory


def benchmark_with_mask(model, batch_size, seq_len_q, seq_len_kv, emb_dim,
                        num_iterations=1000):
    """Benchmark với padding mask"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    query = torch.randn(batch_size, seq_len_q, emb_dim, device=device)
    key_value = torch.randn(batch_size, seq_len_kv, emb_dim, device=device)

    # Create padding mask
    key_padding_mask = torch.randint(0, 2, (batch_size, seq_len_kv),
                                     dtype=torch.bool, device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                if isinstance(model, (OptimizedCrossMultiHeadAttention,
                                      FusedCrossAttention, PyTorchNativeCrossAttention)):
                    output = model(query, key_value, key_padding_mask=key_padding_mask)
                else:
                    output = model(query, key_value)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(query, key_value)
        avg_time = ((time.time() - start_time) * 1000) / num_iterations

    return avg_time


def run_comprehensive_benchmark():
    """Benchmark toàn diện"""
    configs = [
        # (batch_size, seq_len_q, seq_len_kv, emb_dim, num_heads)
        (32, 64, 128, 512, 8),
        (16, 128, 256, 512, 8),
        (8, 256, 512, 512, 8),
        (4, 512, 1024, 512, 8),
        (32, 64, 64, 768, 12),  # BERT-like
    ]

    print("=" * 140)
    print(f"{'Config':<25} {'Model':<35} {'Params (M)':<12} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
    print("=" * 140)

    for batch_size, seq_len_q, seq_len_kv, emb_dim, num_heads in configs:
        config_str = f"B{batch_size}_Q{seq_len_q}_KV{seq_len_kv}_D{emb_dim}"

        models = {
            'Original': OriginalCrossMultiHeadAttention(emb_dim, num_heads),
            'Optimized': OptimizedCrossMultiHeadAttention(emb_dim, num_heads),
            'Fused (Flash Attn)': FusedCrossAttention(emb_dim, num_heads),
            'Efficient (KV-Cache)': EfficientCrossAttention(emb_dim, num_heads),
            'Linear Attention': LinearAttention(emb_dim, num_heads),
            'PyTorch Native': PyTorchNativeCrossAttention(emb_dim, num_heads),
        }

        results = {}

        for name, model in models.items():
            try:
                params = count_parameters(model) / 1e6
                avg_time, memory = benchmark_cross_attention(
                    model, batch_size, seq_len_q, seq_len_kv, emb_dim,
                    num_iterations=1000
                )
                results[name] = {
                    'params': params,
                    'time': avg_time,
                    'memory': memory
                }
            except Exception as e:
                print(f"{config_str:<25} {name:<35} ERROR: {str(e)}")
                results[name] = {
                    'params': 0,
                    'time': float('inf'),
                    'memory': 0
                }

        # Calculate speedup vs Original
        baseline_time = results['Original']['time']

        for name, res in results.items():
            speedup = baseline_time / res['time'] if res['time'] > 0 else 0
            print(f"{config_str:<25} {name:<35} {res['params']:>8.2f} M   "
                  f"{res['time']:>8.4f} ms   {res['memory']:>8.2f} MB   {speedup:>6.2f}x")

        print("-" * 140)


def benchmark_sequence_lengths():
    """Benchmark với các độ dài sequence khác nhau"""
    batch_size, emb_dim, num_heads = 16, 512, 8
    seq_lengths = [(64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]

    print("\n" + "=" * 120)
    print("SEQUENCE LENGTH SCALING")
    print("=" * 120)
    print(f"{'Seq Lengths':<20} {'Original':<15} {'Optimized':<15} {'Linear Attn':<15} {'Speedup':<10}")
    print("=" * 120)

    for seq_len_q, seq_len_kv in seq_lengths:
        seq_str = f"Q{seq_len_q}_KV{seq_len_kv}"

        models = {
            'Original': OriginalCrossMultiHeadAttention(emb_dim, num_heads),
            'Optimized': OptimizedCrossMultiHeadAttention(emb_dim, num_heads),
            'Linear': LinearAttention(emb_dim, num_heads),
        }

        times = {}
        for name, model in models.items():
            try:
                avg_time, _ = benchmark_cross_attention(
                    model, batch_size, seq_len_q, seq_len_kv, emb_dim,
                    num_iterations=100
                )
                times[name] = avg_time
            except Exception as e:
                times[name] = float('inf')

        speedup = times['Original'] / times['Optimized'] if times['Optimized'] > 0 else 0

        print(f"{seq_str:<20} {times['Original']:>10.4f} ms   "
              f"{times['Optimized']:>10.4f} ms   {times['Linear']:>10.4f} ms   "
              f"{speedup:>6.2f}x")


def test_correctness():
    """Kiểm tra tính đúng đắn"""
    batch_size, seq_len_q, seq_len_kv = 2, 8, 16
    emb_dim, num_heads = 512, 8

    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len_q, emb_dim)
    key_value = torch.randn(batch_size, seq_len_kv, emb_dim)

    models = {
        'Original': OriginalCrossMultiHeadAttention(emb_dim, num_heads),
        'Optimized': OptimizedCrossMultiHeadAttention(emb_dim, num_heads, dropout=0.0),
        'Fused': FusedCrossAttention(emb_dim, num_heads, dropout=0.0),
        'PyTorch Native': PyTorchNativeCrossAttention(emb_dim, num_heads, dropout=0.0),
    }

    print("\n" + "=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)

    outputs = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            outputs[name] = model(query, key_value)

        print(f"\n{name}:")
        print(f"  Output shape: {outputs[name].shape}")
        print(f"  Output mean:  {outputs[name].mean().item():.6f}")
        print(f"  Output std:   {outputs[name].std().item():.6f}")
        print(f"  Output min:   {outputs[name].min().item():.6f}")
        print(f"  Output max:   {outputs[name].max().item():.6f}")


def test_with_masks():
    """Test với padding masks"""
    batch_size, seq_len_q, seq_len_kv = 4, 16, 32
    emb_dim, num_heads = 512, 8

    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len_q, emb_dim)
    key_value = torch.randn(batch_size, seq_len_kv, emb_dim)

    # Create padding mask (True = ignore)
    key_padding_mask = torch.zeros(batch_size, seq_len_kv, dtype=torch.bool)
    key_padding_mask[0, 20:] = True  # Mask last 12 positions of first sample
    key_padding_mask[1, 25:] = True  # Mask last 7 positions of second sample

    print("\n" + "=" * 80)
    print("PADDING MASK TEST")
    print("=" * 80)

    models = {
        'Optimized': OptimizedCrossMultiHeadAttention(emb_dim, num_heads, dropout=0.0),
        'PyTorch Native': PyTorchNativeCrossAttention(emb_dim, num_heads, dropout=0.0),
    }

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(query, key_value, key_padding_mask=key_padding_mask)

        print(f"\n{name}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Sample 0 (masked 20:) mean: {output[0].mean().item():.6f}")
        print(f"  Sample 1 (masked 25:) mean: {output[1].mean().item():.6f}")
        print(f"  Sample 2 (no mask) mean: {output[2].mean().item():.6f}")


def benchmark_kv_cache():
    """Benchmark KV caching for inference"""
    batch_size, seq_len_q, seq_len_kv = 1, 1, 512  # Typical for autoregressive decoding
    emb_dim, num_heads = 512, 8

    print("\n" + "=" * 100)
    print("KV-CACHE BENCHMARK (Autoregressive Decoding)")
    print("=" * 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Without cache
    model_no_cache = OptimizedCrossMultiHeadAttention(emb_dim, num_heads).to(device)
    model_no_cache.eval()

    # With cache
    model_with_cache = EfficientCrossAttention(emb_dim, num_heads, use_kv_cache=True).to(device)
    model_with_cache.eval()

    key_value = torch.randn(batch_size, seq_len_kv, emb_dim, device=device)

    # Simulate autoregressive decoding
    num_decode_steps = 100

    # Without cache
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            for _ in range(num_decode_steps):
                query = torch.randn(batch_size, 1, emb_dim, device=device)
                _ = model_no_cache(query, key_value)
        end.record()
        torch.cuda.synchronize()
        time_no_cache = start.elapsed_time(end)
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_decode_steps):
                query = torch.randn(batch_size, 1, emb_dim, device=device)
                _ = model_no_cache(query, key_value)
        time_no_cache = (time.time() - start_time) * 1000

    # With cache
    model_with_cache.clear_cache()

    if device.type == 'cuda':
        start.record()
        with torch.no_grad():
            # First step to build cache
            query = torch.randn(batch_size, 1, emb_dim, device=device)
            _ = model_with_cache(query, key_value, use_cache=True)

            # Subsequent steps use cache
            for _ in range(num_decode_steps - 1):
                query = torch.randn(batch_size, 1, emb_dim, device=device)
                _ = model_with_cache(query, key_value, use_cache=True)
        end.record()
        torch.cuda.synchronize()
        time_with_cache = start.elapsed_time(end)
    else:
        start_time = time.time()
        with torch.no_grad():
            query = torch.randn(batch_size, 1, emb_dim, device=device)
            _ = model_with_cache(query, key_value, use_cache=True)

            for _ in range(num_decode_steps - 1):
                query = torch.randn(batch_size, 1, emb_dim, device=device)
                _ = model_with_cache(query, key_value, use_cache=True)
        time_with_cache = (time.time() - start_time) * 1000

    print(f"Without KV-cache: {time_no_cache:.2f} ms")
    print(f"With KV-cache:    {time_with_cache:.2f} ms")
    print(f"Speedup:          {time_no_cache / time_with_cache:.2f}x")


def profile_complexity():
    """Phân tích computational complexity"""
    batch_size, emb_dim, num_heads = 16, 512, 8
    seq_lengths = [64, 128, 256, 512, 1024]

    print("\n" + "=" * 100)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 100)
    print(f"{'Seq Length':<15} {'Standard Attn':<20} {'Linear Attn':<20} {'Ratio':<10}")
    print("=" * 100)

    for seq_len in seq_lengths:
        model_std = OptimizedCrossMultiHeadAttention(emb_dim, num_heads)
        model_linear = LinearAttention(emb_dim, num_heads)

        time_std, _ = benchmark_cross_attention(
            model_std, batch_size, seq_len, seq_len, emb_dim, num_iterations=100
        )
        time_linear, _ = benchmark_cross_attention(
            model_linear, batch_size, seq_len, seq_len, emb_dim, num_iterations=100
        )

        ratio = time_std / time_linear
        print(f"{seq_len:<15} {time_std:>15.4f} ms   {time_linear:>15.4f} ms   {ratio:>6.2f}x")


def memory_profiling():
    """Profile memory usage"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return

    batch_size, seq_len_q, seq_len_kv = 32, 256, 512
    emb_dim, num_heads = 512, 8
    device = torch.device('cuda')

    models = {
        'Original': OriginalCrossMultiHeadAttention(emb_dim, num_heads),
        'Optimized': OptimizedCrossMultiHeadAttention(emb_dim, num_heads),
        'Fused': FusedCrossAttention(emb_dim, num_heads),
        'Linear': LinearAttention(emb_dim, num_heads),
    }

    print("\n" + "=" * 100)
    print("MEMORY PROFILING")
    print("=" * 100)
    print(f"{'Model':<25} {'Params (MB)':<15} {'Activations (MB)':<18} {'Total (MB)':<15}")
    print("=" * 100)

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        # Parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2

        # Measure activation memory
        torch.cuda.reset_peak_memory_stats()
        query = torch.randn(batch_size, seq_len_q, emb_dim, device=device)
        key_value = torch.randn(batch_size, seq_len_kv, emb_dim, device=device)

        with torch.no_grad():
            output = model(query, key_value)

        torch.cuda.synchronize()
        total_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
        activation_memory = total_memory - param_memory

        print(f"{name:<25} {param_memory:>10.2f} MB   {activation_memory:>12.2f} MB   {total_memory:>10.2f} MB")


if __name__ == "__main__":
    print("=" * 80)
    print("CROSS-ATTENTION OPTIMIZATION BENCHMARK")
    print("=" * 80)

    # Test correctness
    test_correctness()

    # Test with masks
    test_with_masks()

    # Comprehensive benchmark
    run_comprehensive_benchmark()

    # Sequence length scaling
    benchmark_sequence_lengths()

    # KV-cache benchmark
    benchmark_kv_cache()

    # Complexity analysis
    profile_complexity()

    # Memory profiling
    if torch.cuda.is_available():
        memory_profiling()
