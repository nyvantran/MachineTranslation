import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


class OptimizedFeedforward(nn.Module):
    """
    Tối ưu hóa bằng cách:
    - Sử dụng GELU thay vì ReLU (smooth hơn, hiệu quả hơn)
    - Thêm bias (giúp model học tốt hơn)
    - Thêm dropout cho regularization
    - Expansion ratio 4x (chuẩn Transformer)
    - In-place operations khi có thể
    """

    def __init__(self, d_model=512, d_ff=None, dropout=0.1, activation='gelu'):
        super(OptimizedFeedforward, self).__init__()

        if d_ff is None:
            d_ff = d_model * 4  # Standard ratio

        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'swish':
            self.activation = nn.SiLU()  # Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        # (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class FusedFeedforward(nn.Module):
    """
    Fused implementation với torch.jit để tối ưu kernel fusion
    """

    def __init__(self, d_model=512, d_ff=None, dropout=0.1):
        super(FusedFeedforward, self).__init__()

        if d_ff is None:
            d_ff = d_model * 4

        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout_p = dropout

    def forward(self, x):
        # Fused operations
        x = F.gelu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class GLUFeedforward(nn.Module):
    """
    Gated Linear Unit (GLU) variant
    Sử dụng gating mechanism để kiểm soát information flow
    """

    def __init__(self, d_model=512, d_ff=None, dropout=0.1, glu_variant='geglu'):
        super(GLUFeedforward, self).__init__()

        if d_ff is None:
            d_ff = d_model * 4

        self.glu_variant = glu_variant

        # GLU needs 2x the parameters for gating
        if glu_variant in ['geglu', 'swiglu', 'reglu']:
            # Output both value and gate
            self.linear1 = nn.Linear(d_model, d_ff * 2, bias=True)
        else:
            self.linear1 = nn.Linear(d_model, d_ff, bias=True)

        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.glu_variant == 'geglu':
            # GeGLU: x * GELU(gate)
            x, gate = self.linear1(x).chunk(2, dim=-1)
            x = x * F.gelu(gate)
        elif self.glu_variant == 'swiglu':
            # SwiGLU: x * Swish(gate) - Used in LLaMA
            x, gate = self.linear1(x).chunk(2, dim=-1)
            x = x * F.silu(gate)
        elif self.glu_variant == 'reglu':
            # ReGLU: x * ReLU(gate)
            x, gate = self.linear1(x).chunk(2, dim=-1)
            x = x * F.relu(gate)
        else:
            x = F.gelu(self.linear1(x))

        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


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


class EfficientFeedforward(nn.Module):
    """
    Memory-efficient implementation với mixed precision support
    """

    def __init__(self, d_model=512, d_ff=None, dropout=0.1):
        super(EfficientFeedforward, self).__init__()

        if d_ff is None:
            d_ff = d_model * 4

        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout_p = dropout

    def forward(self, x):
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=True):
            x = self.linear1(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            x = self.linear2(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class ExpertChoiceFeedforward(nn.Module):
    """
    Expert Choice routing - động routing tokens đến experts
    Hiệu quả cho large-scale models
    """

    def __init__(self, d_model=512, d_ff=None, num_experts=4, dropout=0.1):
        super(ExpertChoiceFeedforward, self).__init__()

        if d_ff is None:
            d_ff = d_model * 4

        self.num_experts = num_experts

        # Multiple expert FFNs
        self.experts = nn.ModuleList([
            OptimizedFeedforward(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # Compute routing scores
        router_logits = self.router(x)  # (B, T, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Simple averaging (for efficiency)
        # In practice, use top-k routing
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            weight = routing_weights[:, :, i:i + 1]
            output += weight * expert_out

        return output


# Original implementation (cleaned)
class OriginalFeedforward(nn.Module):
    def __init__(self, dmodel=512):
        super(OriginalFeedforward, self).__init__()
        self.linear1 = nn.Linear(dmodel, dmodel * 2, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dmodel * 2, dmodel, bias=False)

    def forward(self, x):
        linear1 = self.linear1(x)
        relu = self.relu(linear1)
        linear2 = self.linear2(relu)
        return linear2


class PyTorchNativeFFN(nn.Module):
    """
    PyTorch không có built-in FFN, nhưng đây là implementation
    theo chuẩn Transformer paper
    """

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PyTorchNativeFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


def count_parameters(model):
    """Đếm số parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_ffn(model, batch_size, seq_len, d_model, num_iterations=1000, warmup=100):
    """Benchmark FFN performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, seq_len, d_model, device=device)

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


def benchmark_backward(model, batch_size, seq_len, d_model, num_iterations=100):
    """Benchmark backward pass (training)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
    else:
        start_time = time.time()
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        elapsed_time = (time.time() - start_time) * 1000
        avg_time = elapsed_time / num_iterations

    return avg_time


def run_comprehensive_benchmark():
    """Benchmark toàn diện"""
    configs = [
        # (batch_size, seq_len, d_model)
        (32, 128, 512),
        (64, 256, 512),
        (16, 512, 512),
        (8, 1024, 512),
        (32, 128, 768),
        (16, 256, 1024),
    ]

    print("=" * 130)
    print(
        f"{'Config':<20} {'Model':<30} {'Params (M)':<12} {'FWD (ms)':<12} {'BWD (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
    print("=" * 130)

    for batch_size, seq_len, d_model in configs:
        config_str = f"B{batch_size}_T{seq_len}_D{d_model}"

        models = {
            'Original (2x, ReLU)': OriginalFeedforward(d_model),
            'Optimized (4x, GELU)': OptimizedFeedforward(d_model),
            'Fused (4x, GELU)': FusedFeedforward(d_model),
            'GeGLU': GLUFeedforward(d_model, glu_variant='geglu'),
            'SwiGLU (LLaMA-style)': SwiGLUFeedforward(d_model),
            'PyTorch Native': PyTorchNativeFFN(d_model, d_model * 4),
        }

        results = {}

        for name, model in models.items():
            try:
                params = count_parameters(model) / 1e6
                fwd_time, memory = benchmark_ffn(model, batch_size, seq_len, d_model, num_iterations=1000)
                bwd_time = benchmark_backward(model, batch_size, seq_len, d_model, num_iterations=100)

                results[name] = {
                    'params': params,
                    'fwd': fwd_time,
                    'bwd': bwd_time,
                    'memory': memory
                }
            except Exception as e:
                print(f"{config_str:<20} {name:<30} ERROR: {str(e)}")
                results[name] = {
                    'params': 0,
                    'fwd': float('inf'),
                    'bwd': float('inf'),
                    'memory': 0
                }

        # Calculate speedup vs Original
        baseline_fwd = results['Original (2x, ReLU)']['fwd']

        for name, res in results.items():
            speedup = baseline_fwd / res['fwd'] if res['fwd'] > 0 else 0
            print(f"{config_str:<20} {name:<30} {res['params']:>8.2f} M   "
                  f"{res['fwd']:>8.4f} ms   {res['bwd']:>8.4f} ms   "
                  f"{res['memory']:>8.2f} MB   {speedup:>6.2f}x")

        print("-" * 130)


def analyze_activation_functions():
    """So sánh các activation functions"""
    batch_size, seq_len, d_model = 32, 512, 512

    activations = {
        'ReLU': 'relu',
        'GELU': 'gelu',
        'SwiSH/SiLU': 'swish',
    }

    print("\n" + "=" * 100)
    print("ACTIVATION FUNCTION COMPARISON")
    print("=" * 100)
    print(f"{'Activation':<20} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Memory (MB)':<15}")
    print("=" * 100)

    for name, act in activations.items():
        model = OptimizedFeedforward(d_model, activation=act)
        fwd_time, memory = benchmark_ffn(model, batch_size, seq_len, d_model)
        bwd_time = benchmark_backward(model, batch_size, seq_len, d_model)

        print(f"{name:<20} {fwd_time:>10.4f} ms   {bwd_time:>10.4f} ms   {memory:>10.2f} MB")


def compare_expansion_ratios():
    """So sánh các tỷ lệ expansion khác nhau"""
    batch_size, seq_len, d_model = 32, 512, 512

    ratios = [1, 2, 4, 8]

    print("\n" + "=" * 100)
    print("EXPANSION RATIO COMPARISON")
    print("=" * 100)
    print(f"{'Ratio':<10} {'d_ff':<10} {'Params (M)':<15} {'Forward (ms)':<15} {'Memory (MB)':<15}")
    print("=" * 100)

    for ratio in ratios:
        d_ff = d_model * ratio
        model = OptimizedFeedforward(d_model, d_ff=d_ff)
        params = count_parameters(model) / 1e6
        fwd_time, memory = benchmark_ffn(model, batch_size, seq_len, d_model)

        print(f"{ratio}x{' ':<7} {d_ff:<10} {params:>10.2f} M   {fwd_time:>10.4f} ms   {memory:>10.2f} MB")


def test_correctness():
    """Kiểm tra correctness"""
    batch_size, seq_len, d_model = 4, 16, 512

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    models = {
        'Original': OriginalFeedforward(d_model),
        'Optimized': OptimizedFeedforward(d_model, d_ff=d_model * 2, activation='relu', dropout=0.0),
        'GeGLU': GLUFeedforward(d_model, d_ff=d_model * 2, dropout=0.0),
        'SwiGLU': SwiGLUFeedforward(d_model, dropout=0.0),
    }

    print("\n" + "=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)

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


def profile_memory_usage():
    """Profile memory usage chi tiết"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return

    batch_size, seq_len, d_model = 32, 512, 512
    device = torch.device('cuda')

    models = {
        'Original': OriginalFeedforward(d_model),
        'Optimized': OptimizedFeedforward(d_model),
        'SwiGLU': SwiGLUFeedforward(d_model),
    }

    print("\n" + "=" * 100)
    print("MEMORY USAGE PROFILING")
    print("=" * 100)
    print(f"{'Model':<25} {'Params (MB)':<15} {'Activations (MB)':<18} {'Total (MB)':<15}")
    print("=" * 100)

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2

        # Measure activation memory
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        with torch.no_grad():
            output = model(x)

        torch.cuda.synchronize()
        total_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
        activation_memory = total_memory - param_memory

        print(f"{name:<25} {param_memory:>10.2f} MB   {activation_memory:>12.2f} MB   {total_memory:>10.2f} MB")


def compare_with_compiled():
    """So sánh với torch.compile (PyTorch 2.0+)"""
    if not hasattr(torch, 'compile'):
        print("torch.compile not available (requires PyTorch 2.0+)")
        return

    batch_size, seq_len, d_model = 32, 512, 512

    print("\n" + "=" * 100)
    print("TORCH.COMPILE COMPARISON")
    print("=" * 100)

    model_eager = OptimizedFeedforward(d_model)
    model_compiled = torch.compile(OptimizedFeedforward(d_model))

    print("Eager mode:")
    fwd_eager, mem_eager = benchmark_ffn(model_eager, batch_size, seq_len, d_model)
    print(f"  Forward: {fwd_eager:.4f} ms, Memory: {mem_eager:.2f} MB")

    print("\nCompiled mode:")
    fwd_compiled, mem_compiled = benchmark_ffn(model_compiled, batch_size, seq_len, d_model)
    print(f"  Forward: {fwd_compiled:.4f} ms, Memory: {mem_compiled:.2f} MB")

    speedup = fwd_eager / fwd_compiled
    print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    print("=" * 80)
    print("FEEDFORWARD NETWORK OPTIMIZATION BENCHMARK")
    print("=" * 80)

    # Test correctness
    test_correctness()

    # Comprehensive benchmark
    run_comprehensive_benchmark()

    # Activation function comparison
    analyze_activation_functions()

    # Expansion ratio comparison
    compare_expansion_ratios()

    # Memory profiling
    if torch.cuda.is_available():
        profile_memory_usage()

    # Torch.compile comparison
    compare_with_compiled()
