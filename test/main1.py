import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


class OptimizedMultiHeadAttention(nn.Module):
    """
    Tối ưu hóa bằng cách:
    - Batch tất cả heads thành 1 phép toán
    - aCache mask và scale fctor
    - Sử dụng tensor operations thay vì list comprehension
    - Loại bỏ các tính toán không cần thiết
    """

    def __init__(self, emb_dim, num_heads=4, dropout=0.1, atmask=False):
        super(OptimizedMultiHeadAttention, self).__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.atmask = atmask

        # Batch all Q, K, V projections into single matrices
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=True)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        # Register scale as buffer (không train, tự động chuyển device)
        self.register_buffer('scale', torch.tensor(1.0 / math.sqrt(self.head_dim)))

        self.dropout = nn.Dropout(dropout)

        # Cache mask
        self.register_buffer('mask', None)
        self.max_seq_len = 0

    def _get_mask(self, seq_len, device):
        """Cache causal mask"""
        if self.mask is None or seq_len > self.max_seq_len:
            self.mask = torch.triu(
                torch.ones((seq_len, seq_len), device=device),
                diagonal=1
            ).bool()
            self.max_seq_len = seq_len
        return self.mask[:seq_len, :seq_len]

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()

        # Single matrix multiplication for Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim) # (B, T, 3, H, D_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, T, D_h)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if self.atmask:
            mask = self._get_mask(seq_len, x.device)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, T, D_h)

        # Concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, T, H, D_h)
        out = out.reshape(batch_size, seq_len, emb_dim)  # (B, T, D)

        out = self.out_proj(out)
        return out


class OptimizedFlashAttention(nn.Module):
    """
    Sử dụng Flash Attention 2 (nếu có) hoặc scaled_dot_product_attention
    PyTorch 2.0+ có built-in Flash Attention
    """

    def __init__(self, emb_dim, num_heads=4, dropout=0.1, atmask=False):
        super(OptimizedFlashAttention, self).__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.atmask = atmask
        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=True)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use PyTorch's optimized scaled_dot_product_attention (Flash Attention)
        # Available in PyTorch 2.0+
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=self.atmask
            )
        else:
            # Fallback to manual implementation
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            if self.atmask:
                mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)
            out = torch.matmul(attn, v)

        # Reshape and project
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(batch_size, seq_len, emb_dim)
        out = self.out_proj(out)

        return out


# Original implementation (đã fix một số lỗi)
class OriginalMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=4, atmask=False):
        super(OriginalMultiHeadAttention, self).__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.atmask = atmask
        self.head_dim = emb_dim // num_heads

        self.heads = nn.ModuleList(
            [HeadAttention(self.emb_dim, self.head_dim, atmask=self.atmask)
             for _ in range(num_heads)])
        self.combo_linear = nn.Linear(self.head_dim * num_heads, emb_dim)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
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
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale.to(query.device)

        if self.atmask:
            batch_size, seq_len, _ = x.size()
            mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask.unsqueeze(0).expand(batch_size, -1, -1), float('-inf'))

        attn = self.softmax(scores)
        output = torch.matmul(attn, value)
        return output


class PyTorchNativeAttention(nn.Module):
    """Wrapper cho nn.MultiheadAttention của PyTorch"""

    def __init__(self, emb_dim, num_heads=4, dropout=0.1, atmask=False):
        super(PyTorchNativeAttention, self).__init__()
        self.atmask = atmask
        self.mha = nn.MultiheadAttention(
            emb_dim,
            num_heads,
            dropout=dropout,
            batch_first=True  # PyTorch 1.9+
        )

    def forward(self, x):
        if self.atmask:
            seq_len = x.size(1)
            mask = torch.triu(
                torch.ones((seq_len, seq_len), device=x.device),
                diagonal=1
            ).bool()
            out, _ = self.mha(x, x, x, attn_mask=mask)
        else:
            out, _ = self.mha(x, x, x)
        return out


def benchmark_attention(model, batch_size, seq_len, emb_dim, num_iterations=100, warmup=10):
    """Đo thời gian và memory usage"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, seq_len, emb_dim, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations * 1000  # ms

    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
    else:
        peak_memory = 0

    return avg_time, peak_memory


def run_comprehensive_benchmark():
    """Chạy benchmark toàn diện"""
    configs = [
        # (batch_size, seq_len, emb_dim, num_heads)
        (32, 128, 512, 8),
        (16, 256, 512, 8),
        (8, 512, 512, 8),
        (4, 1024, 512, 8),
    ]

    print("=" * 100)
    print(f"{'Config':<20} {'Model':<25} {'Time (ms)':<15} {'Memory (MB)':<15}")
    print("=" * 100)

    for batch_size, seq_len, emb_dim, num_heads in configs:
        config_str = f"B{batch_size}_T{seq_len}_D{emb_dim}"

        models = {
            'Original': OriginalMultiHeadAttention(emb_dim, num_heads, atmask=True),
            'Optimized': OptimizedMultiHeadAttention(emb_dim, num_heads, atmask=True),
            'Flash Attention': OptimizedFlashAttention(emb_dim, num_heads, atmask=True),
            'PyTorch Native': PyTorchNativeAttention(emb_dim, num_heads, atmask=True),
        }

        for name, model in models.items():
            try:
                avg_time, peak_memory = benchmark_attention(
                    model, batch_size, seq_len, emb_dim, num_iterations=100
                )
                print(f"{config_str:<20} {name:<25} {avg_time:>10.3f} ms   {peak_memory:>10.2f} MB")
            except Exception as e:
                print(f"{config_str:<20} {name:<25} ERROR: {str(e)}")

        print("-" * 100)


def test_correctness():
    """Kiểm tra tính đúng đắn của các implementation"""
    batch_size, seq_len, emb_dim, num_heads = 2, 8, 512, 8

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, emb_dim)

    models = {
        'Original': OriginalMultiHeadAttention(emb_dim, num_heads, atmask=True),
        'Optimized': OptimizedMultiHeadAttention(emb_dim, num_heads, atmask=True),
    }

    print("\n" + "=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    outputs = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            outputs[name] = model(x)
        print(f"{name:<20} Output shape: {outputs[name].shape}")
        print(f"{name:<20} Output mean: {outputs[name].mean().item():.6f}")
        print(f"{name:<20} Output std: {outputs[name].std().item():.6f}")
        print("-" * 60)


if __name__ == "__main__":
    print("Testing correctness...")
    test_correctness()

    print("\n\nRunning comprehensive benchmark...")
    run_comprehensive_benchmark()
