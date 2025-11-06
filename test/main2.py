import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


class OptimizedEmbedding(nn.Module):
    """
    Tối ưu hóa bằng cách:
    - Cache positional encoding (không tính lại mỗi forward)
    - Sử dụng register_buffer để tự động chuyển device
    - Broadcasting thay vì repeat
    - Pre-compute trong __init__
    - Thêm dropout cho regularization
    """

    def __init__(self, vocab_size, emb_dim=512, max_seq_len=5000, dropout=0.1):
        super(OptimizedEmbedding, self).__init__()
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


class UltraOptimizedEmbedding(nn.Module):
    """
    Sử dụng learned positional embedding thay vì sinusoidal
    Nhanh hơn và đôi khi hiệu quả hơn
    """

    def __init__(self, vocab_size, emb_dim=512, max_seq_len=5000, dropout=0.1):
        super(UltraOptimizedEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        # Learned positional embedding (như BERT)
        self.pos_emb = nn.Embedding(max_seq_len, emb_dim)
        self.dropout = nn.Dropout(dropout)

        # Scaling factor (như trong Transformer paper)
        self.scale = math.sqrt(emb_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - token indices
        """
        batch_size, seq_len = x.size()

        # Token embedding với scaling
        token_emb = self.token_emb(x) * self.scale

        # Position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.pos_emb(positions)  # (1, seq_len, emb_dim)

        # Combine
        x = token_emb + pos_emb

        return self.dropout(x)


class FusedEmbedding(nn.Module):
    """
    Fused implementation - tối ưu nhất cho inference
    """

    def __init__(self, vocab_size, emb_dim=512, max_seq_len=5000, dropout=0.1):
        super(FusedEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_p = dropout

        # Single embedding lookup table
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # Pre-compute và cache positional encoding
        pe = torch.zeros(max_seq_len, emb_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    @torch.jit.script_method
    def forward(self, x):
        seq_len = x.size(1)

        # Fused operation
        emb = self.embedding(x)
        emb = emb + self.pe[:seq_len, :]

        if self.training:
            emb = F.dropout(emb, p=self.dropout_p, training=True)

        return emb


# Original implementation (cleaned up)
class OriginalEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim=512):
        super(OriginalEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)

    def token_embedding(self, x):
        return self.embedding(x)

    def position_embedding(self, x):
        batch_size, seq_len, emb_dim = x.size()
        pe = torch.zeros(seq_len, emb_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
        return pe

    def forward(self, x):
        emd = self.token_embedding(x)
        pos = self.position_embedding(emd).to(x.device)
        return torch.add(emd, pos)


class PyTorchNativeEmbedding(nn.Module):
    """Wrapper sử dụng PyTorch built-in components"""

    def __init__(self, vocab_size, emb_dim=512, max_seq_len=5000, dropout=0.1):
        super(PyTorchNativeEmbedding, self).__init__()

        # PyTorch's embedding
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # Learned positional embedding
        self.position_embedding = nn.Embedding(max_seq_len, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Embed tokens and positions
        token_emb = self.embedding(x)
        pos_emb = self.position_embedding(positions)

        # Combine
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)

        return embeddings


def benchmark_embedding(model, batch_size, seq_len, vocab_size, num_iterations=1000, warmup=100):
    """Benchmark embedding layers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Random token indices
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(x)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(x)
        elapsed_time = (time.time() - start_time) * 1000
        avg_time = elapsed_time / num_iterations
        peak_memory = 0

    return avg_time, peak_memory


def benchmark_memory_usage(model, batch_size, seq_len, vocab_size):
    """Đo memory usage chi tiết"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

        # Forward pass
        with torch.no_grad():
            output = model(x)

        torch.cuda.synchronize()

        model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 ** 2
        activation_memory = torch.cuda.max_memory_allocated() / 1024 ** 2 - model_memory - buffer_memory

        return {
            'model': model_memory,
            'buffers': buffer_memory,
            'activation': activation_memory,
            'total': model_memory + buffer_memory + activation_memory
        }
    else:
        return {'model': 0, 'buffers': 0, 'activation': 0, 'total': 0}


def run_comprehensive_benchmark():
    """Benchmark toàn diện"""
    configs = [
        # (batch_size, seq_len, vocab_size, emb_dim)
        (32, 128, 10000, 512),
        (64, 256, 10000, 512),
        (16, 512, 10000, 512),
        (8, 1024, 10000, 512),
        (128, 64, 50000, 768),
    ]

    print("=" * 120)
    print(f"{'Config':<25} {'Model':<25} {'Time (ms)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
    print("=" * 120)

    for batch_size, seq_len, vocab_size, emb_dim in configs:
        config_str = f"B{batch_size}_T{seq_len}_V{vocab_size // 1000}k"

        models = {
            'Original': OriginalEmbedding(vocab_size, emb_dim),
            'Optimized': OptimizedEmbedding(vocab_size, emb_dim),
            'Ultra-Optimized': UltraOptimizedEmbedding(vocab_size, emb_dim),
            'PyTorch Native': PyTorchNativeEmbedding(vocab_size, emb_dim),
        }

        results = {}

        for name, model in models.items():
            try:
                avg_time, peak_memory = benchmark_embedding(
                    model, batch_size, seq_len, vocab_size, num_iterations=1000
                )
                results[name] = {'time': avg_time, 'memory': peak_memory}
            except Exception as e:
                print(f"{config_str:<25} {name:<25} ERROR: {str(e)}")
                results[name] = {'time': float('inf'), 'memory': 0}

        # Calculate speedup vs Original
        baseline_time = results['Original']['time']

        for name, res in results.items():
            speedup = baseline_time / res['time'] if res['time'] > 0 else 0
            print(f"{config_str:<25} {name:<25} {res['time']:>10.4f} ms   "
                  f"{res['memory']:>10.2f} MB   {speedup:>6.2f}x")

        print("-" * 120)


def detailed_memory_analysis():
    """Phân tích memory chi tiết"""
    batch_size, seq_len, vocab_size, emb_dim = 32, 512, 10000, 512

    models = {
        'Original': OriginalEmbedding(vocab_size, emb_dim),
        'Optimized': OptimizedEmbedding(vocab_size, emb_dim),
        'Ultra-Optimized': UltraOptimizedEmbedding(vocab_size, emb_dim),
    }

    print("\n" + "=" * 100)
    print("DETAILED MEMORY ANALYSIS")
    print("=" * 100)
    print(f"{'Model':<25} {'Parameters':<15} {'Buffers':<15} {'Activation':<15} {'Total':<15}")
    print("=" * 100)

    for name, model in models.items():
        mem = benchmark_memory_usage(model, batch_size, seq_len, vocab_size)
        print(f"{name:<25} {mem['model']:>10.2f} MB   {mem['buffers']:>10.2f} MB   "
              f"{mem['activation']:>10.2f} MB   {mem['total']:>10.2f} MB")


def test_correctness():
    """Kiểm tra output correctness"""
    batch_size, seq_len, vocab_size, emb_dim = 4, 16, 1000, 512

    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    models = {
        'Original': OriginalEmbedding(vocab_size, emb_dim),
        'Optimized': OptimizedEmbedding(vocab_size, emb_dim),
        'Ultra-Optimized': UltraOptimizedEmbedding(vocab_size, emb_dim),
    }

    print("\n" + "=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)

    # Set same weights for token embedding
    with torch.no_grad():
        base_weight = torch.randn(vocab_size, emb_dim)
        for model in models.values():
            if hasattr(model, 'token_emb'):
                model.token_emb.weight.copy_(base_weight)
            elif hasattr(model, 'embedding'):
                model.embedding.weight.copy_(base_weight)

    outputs = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            outputs[name] = model(x)

        print(f"\n{name}:")
        print(f"  Output shape: {outputs[name].shape}")
        print(f"  Output mean:  {outputs[name].mean().item():.6f}")
        print(f"  Output std:   {outputs[name].std().item():.6f}")
        print(f"  Output min:   {outputs[name].min().item():.6f}")
        print(f"  Output max:   {outputs[name].max().item():.6f}")

    # Compare Original vs Optimized (should be very close for sinusoidal)
    if 'Original' in outputs and 'Optimized' in outputs:
        diff = (outputs['Original'] - outputs['Optimized']).abs().max().item()
        print(f"\nMax difference (Original vs Optimized): {diff:.10f}")


def profile_forward_pass():
    """Profile từng bước trong forward pass"""
    batch_size, seq_len, vocab_size, emb_dim = 32, 512, 10000, 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 80)
    print("FORWARD PASS PROFILING")
    print("=" * 80)

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Original
    print("\nOriginal Implementation:")
    model = OriginalEmbedding(vocab_size, emb_dim).to(device)
    model.eval()

    if device.type == 'cuda':
        with torch.no_grad():
            torch.cuda.synchronize()

            # Token embedding
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            token_emb = model.token_embedding(x)
            end.record()
            torch.cuda.synchronize()
            print(f"  Token embedding: {start.elapsed_time(end):.4f} ms")

            # Position embedding
            start.record()
            pos_emb = model.position_embedding(token_emb)
            end.record()
            torch.cuda.synchronize()
            print(f"  Position embedding: {start.elapsed_time(end):.4f} ms")

            # Add
            start.record()
            output = torch.add(token_emb, pos_emb.to(x.device))
            end.record()
            torch.cuda.synchronize()
            print(f"  Addition: {start.elapsed_time(end):.4f} ms")

    # Optimized
    print("\nOptimized Implementation:")
    model = OptimizedEmbedding(vocab_size, emb_dim).to(device)
    model.eval()

    if device.type == 'cuda':
        with torch.no_grad():
            # Token embedding
            start.record()
            token_emb = model.token_emb(x)
            end.record()
            torch.cuda.synchronize()
            print(f"  Token embedding: {start.elapsed_time(end):.4f} ms")

            # Position lookup (cached)
            start.record()
            pos_emb = model.pe[:, :seq_len, :]
            end.record()
            torch.cuda.synchronize()
            print(f"  Position lookup: {start.elapsed_time(end):.4f} ms")

            # Add
            start.record()
            output = token_emb + pos_emb
            end.record()
            torch.cuda.synchronize()
            print(f"  Addition: {start.elapsed_time(end):.4f} ms")


if __name__ == "__main__":
    # print("=" * 80)
    # print("EMBEDDING LAYER OPTIMIZATION BENCHMARK")
    # print("=" * 80)
    #
    # # Test correctness
    # test_correctness()
    #
    # # Run comprehensive benchmark
    # run_comprehensive_benchmark()
    #
    # # Detailed memory analysis
    # if torch.cuda.is_available():
    #     detailed_memory_analysis()
    #     profile_forward_pass()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    embeding = OptimizedEmbedding(vocab_size=10000, emb_dim=512)
    embeding = embeding.to(device)
    embeding.train()
    x = torch.randint(0, 10000, (32, 128)).to(device)
    output = embeding(x)
    print(output.shape)
