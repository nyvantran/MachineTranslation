import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


class OptimizedCrossEntropyLoss(nn.Module):
    """
    Optimized Cross Entropy Loss với:
    - Ignore padding tokens
    - Label smoothing
    - Efficient computation
    - Multiple metrics
    """

    def __init__(
            self,
            vocab_size: int,
            pad_idx: int = 0,
            label_smoothing: float = 0.1,
            reduction: str = 'mean',
            ignore_index: Optional[int] = None
    ):
        super(OptimizedCrossEntropyLoss, self).__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index if ignore_index is not None else pad_idx

        # Use built-in CrossEntropyLoss with optimizations
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction
        )

    def forward(
            self,
            predict: torch.Tensor,
            target: torch.Tensor,
            return_metrics: bool = False
    ) -> torch.Tensor:
        """
        Args:
            predict: (batch_size, seq_len, vocab_size)
            target: (batch_size, seq_len)
            return_metrics: whether to return additional metrics
        Returns:
            loss or (loss, metrics_dict)
        """
        batch_size, seq_len, vocab_size = predict.size()

        # Reshape efficiently
        predict_flat = predict.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        # Compute loss
        loss = self.loss_fn(predict_flat, target_flat)

        if return_metrics:
            with torch.no_grad():
                metrics = self._compute_metrics(predict, target, predict_flat, target_flat)
            return loss, metrics

        return loss

    def _compute_metrics(
            self,
            predict: torch.Tensor,
            target: torch.Tensor,
            predict_flat: torch.Tensor,
            target_flat: torch.Tensor
    ) -> Dict[str, float]:
        """Compute additional metrics"""
        # Mask for non-padding tokens
        mask = (target_flat != self.ignore_index)

        # Accuracy
        pred_labels = predict_flat.argmax(dim=-1)
        correct = (pred_labels == target_flat) & mask
        accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0

        # Perplexity
        with torch.cuda.amp.autocast(enabled=False):
            log_probs = F.log_softmax(predict_flat.float(), dim=-1)
            nll_loss = F.nll_loss(
                log_probs,
                target_flat,
                ignore_index=self.ignore_index,
                reduction='mean'
            )
            perplexity = torch.exp(nll_loss).item()

        # Token count
        num_tokens = mask.sum().item()

        return {
            'accuracy': accuracy,
            'perplexity': perplexity,
            'num_tokens': num_tokens
        }


class LabelSmoothingLoss(nn.Module):
    """
    Custom Label Smoothing Loss với KL Divergence
    Hiệu quả hơn built-in cho một số trường hợp
    """

    def __init__(
            self,
            vocab_size: int,
            pad_idx: int = 0,
            smoothing: float = 0.1,
            reduction: str = 'mean'
    ):
        super(LabelSmoothingLoss, self).__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predict: (batch_size, seq_len, vocab_size) - logits
            target: (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = predict.size()

        # Reshape
        predict_flat = predict.reshape(-1, vocab_size)  # (N, V)
        target_flat = target.reshape(-1)  # (N,)

        # Log softmax
        log_probs = F.log_softmax(predict_flat, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (vocab_size - 2))  # -2 for true label and padding
            true_dist.scatter_(1, target_flat.unsqueeze(1), self.confidence)

            # Zero out padding
            true_dist[:, self.pad_idx] = 0
            mask = (target_flat == self.pad_idx)
            if mask.any():
                true_dist[mask] = 0.0

        # KL divergence
        loss = -torch.sum(true_dist * log_probs, dim=-1)

        # Apply mask
        mask = (target_flat != self.pad_idx)
        loss = loss * mask.float()

        if self.reduction == 'mean':
            return loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.reshape(batch_size, seq_len)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced translation tasks
    Focuses on hard examples
    """

    def __init__(
            self,
            vocab_size: int,
            pad_idx: int = 0,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predict: (batch_size, seq_len, vocab_size)
            target: (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = predict.size()

        # Reshape
        predict_flat = predict.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        # Probabilities
        probs = F.softmax(predict_flat, dim=-1)

        # Get probabilities of true class
        target_probs = probs.gather(1, target_flat.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - target_probs) ** self.gamma

        # Cross entropy
        log_probs = F.log_softmax(predict_flat, dim=-1)
        ce_loss = F.nll_loss(log_probs, target_flat, reduction='none')

        # Focal loss
        loss = self.alpha * focal_weight * ce_loss

        # Mask padding
        mask = (target_flat != self.pad_idx)
        loss = loss * mask.float()

        if self.reduction == 'mean':
            return loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.reshape(batch_size, seq_len)


class TranslationLoss(nn.Module):
    """
    Complete translation loss với multiple objectives
    """

    def __init__(
            self,
            vocab_size: int,
            pad_idx: int = 0,
            label_smoothing: float = 0.1,
            length_penalty: float = 0.0,
            coverage_penalty: float = 0.0
    ):
        super(TranslationLoss, self).__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty

        # Main loss
        self.ce_loss = OptimizedCrossEntropyLoss(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            label_smoothing=label_smoothing
        )

    def forward(
            self,
            predict: torch.Tensor,
            target: torch.Tensor,
            src_lengths: Optional[torch.Tensor] = None,
            tgt_lengths: Optional[torch.Tensor] = None,
            attention_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predict: (batch_size, seq_len, vocab_size)
            target: (batch_size, seq_len)
            src_lengths: (batch_size,)
            tgt_lengths: (batch_size,)
            attention_weights: (batch_size, tgt_len, src_len)
        """
        # Main cross-entropy loss
        ce_loss, metrics = self.ce_loss(predict, target, return_metrics=True)

        total_loss = ce_loss

        # Length penalty (encourage similar lengths)
        if self.length_penalty > 0 and src_lengths is not None and tgt_lengths is not None:
            length_diff = (tgt_lengths.float() - src_lengths.float()).abs()
            length_loss = length_diff.mean()
            total_loss = total_loss + self.length_penalty * length_loss
            metrics['length_loss'] = length_loss.item()

        # Coverage penalty (prevent over/under attention)
        if self.coverage_penalty > 0 and attention_weights is not None:
            # Sum attention over target positions
            coverage = attention_weights.sum(dim=1)  # (batch_size, src_len)
            # Penalty for deviation from 1.0
            coverage_loss = ((coverage - 1.0) ** 2).mean()
            total_loss = total_loss + self.coverage_penalty * coverage_loss
            metrics['coverage_loss'] = coverage_loss.item()

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple objectives
    """

    def __init__(
            self,
            vocab_size: int,
            pad_idx: int = 0,
            label_smoothing: float = 0.1,
            use_focal: bool = False,
            focal_gamma: float = 2.0,
            kl_weight: float = 0.0,
            auxiliary_weight: float = 0.0
    ):
        super(CompositeLoss, self).__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.kl_weight = kl_weight
        self.auxiliary_weight = auxiliary_weight

        # Main loss
        if use_focal:
            self.main_loss = FocalLoss(vocab_size, pad_idx, gamma=focal_gamma)
        else:
            self.main_loss = OptimizedCrossEntropyLoss(
                vocab_size, pad_idx, label_smoothing=label_smoothing
            )

        # Optional KL divergence (for knowledge distillation)
        if kl_weight > 0:
            self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
            self,
            predict: torch.Tensor,
            target: torch.Tensor,
            teacher_logits: Optional[torch.Tensor] = None,
            auxiliary_logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predict: (batch_size, seq_len, vocab_size) - student logits
            target: (batch_size, seq_len)
            teacher_logits: (batch_size, seq_len, vocab_size) - for distillation
            auxiliary_logits: (batch_size, seq_len, vocab_size) - auxiliary predictions
        """
        metrics = {}

        # Main loss
        if isinstance(self.main_loss, OptimizedCrossEntropyLoss):
            main_loss, main_metrics = self.main_loss(predict, target, return_metrics=True)
            metrics.update(main_metrics)
        else:
            main_loss = self.main_loss(predict, target)

        total_loss = main_loss
        metrics['main_loss'] = main_loss.item()

        # KL divergence with teacher (knowledge distillation)
        if self.kl_weight > 0 and teacher_logits is not None:
            student_log_probs = F.log_softmax(predict, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            kl_div = self.kl_loss(student_log_probs, teacher_probs)
            total_loss = total_loss + self.kl_weight * kl_div
            metrics['kl_loss'] = kl_div.item()

        # Auxiliary loss (e.g., from intermediate layers)
        if self.auxiliary_weight > 0 and auxiliary_logits is not None:
            aux_loss = self.main_loss(auxiliary_logits, target)
            total_loss = total_loss + self.auxiliary_weight * aux_loss
            metrics['auxiliary_loss'] = aux_loss.item()

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


# Original implementation
class OriginalCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.0, len_vocab=None):
        super(OriginalCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.len_vocab = len_vocab if len_vocab is not None else 1000
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

    def forward(self, predict, target):
        batch_size, seq_len, vocab_size = predict.size()
        predict = predict.view(-1, vocab_size)
        target = target.view(-1)
        loss = self.loss_fn(predict, target)
        return loss


# PyTorch Native (wrapper)
class PyTorchNativeLoss(nn.Module):
    """Wrapper for PyTorch's built-in CrossEntropyLoss"""

    def __init__(self, vocab_size: int, pad_idx: int = 0, label_smoothing: float = 0.1):
        super(PyTorchNativeLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
            reduction='mean'
        )

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, vocab_size = predict.size()
        predict_flat = predict.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)
        return self.loss_fn(predict_flat, target_flat)


import time
import numpy as np


def benchmark_loss(loss_fn, predict, target, num_iterations=1000, warmup=100):
    """Benchmark loss computation"""
    device = predict.device

    # Warmup
    for _ in range(warmup):
        loss = loss_fn(predict, target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            loss = loss_fn(predict, target)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
    else:
        start_time = time.time()
        for _ in range(num_iterations):
            loss = loss_fn(predict, target)
        elapsed_time = (time.time() - start_time) * 1000
        avg_time = elapsed_time / num_iterations

    return avg_time


def benchmark_backward(loss_fn, predict, target, num_iterations=100):
    """Benchmark backward pass"""
    device = predict.device

    # Warmup
    for _ in range(10):
        predict_copy = predict.clone().requires_grad_(True)
        loss = loss_fn(predict_copy, target)
        loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            predict_copy = predict.clone().requires_grad_(True)
            loss = loss_fn(predict_copy, target)
            loss.backward()
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
    else:
        start_time = time.time()
        for _ in range(num_iterations):
            predict_copy = predict.clone().requires_grad_(True)
            loss = loss_fn(predict_copy, target)
            loss.backward()
        elapsed_time = (time.time() - start_time) * 1000
        avg_time = elapsed_time / num_iterations

    return avg_time


def run_comprehensive_benchmark():
    """Comprehensive benchmark"""
    configs = [
        # (batch_size, seq_len, vocab_size)
        (32, 64, 10000),
        (16, 128, 10000),
        (8, 256, 10000),
        (32, 64, 50000),
    ]

    print("=" * 120)
    print(f"{'Config':<20} {'Loss Function':<30} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15}")
    print("=" * 120)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for batch_size, seq_len, vocab_size in configs:
        config_str = f"B{batch_size}_T{seq_len}_V{vocab_size // 1000}k"

        # Create sample data
        predict = torch.randn(batch_size, seq_len, vocab_size, device=device)
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Losses to test
        losses = {
            'Original': OriginalCrossEntropyLoss(len_vocab=vocab_size, smoothing=0.1),
            'Optimized': OptimizedCrossEntropyLoss(vocab_size, label_smoothing=0.1),
            'Label Smoothing': LabelSmoothingLoss(vocab_size, smoothing=0.1),
            'Focal Loss': FocalLoss(vocab_size),
            'Translation Loss': TranslationLoss(vocab_size, label_smoothing=0.1),
            'PyTorch Native': PyTorchNativeLoss(vocab_size, label_smoothing=0.1),
        }

        for name, loss_fn in losses.items():
            try:
                loss_fn = loss_fn.to(device)

                # Forward pass
                fwd_time = benchmark_loss(loss_fn, predict, target, num_iterations=1000)

                # Backward pass
                bwd_time = benchmark_backward(loss_fn, predict, target, num_iterations=100)

                total_time = fwd_time + bwd_time

                print(
                    f"{config_str:<20} {name:<30} {fwd_time:>10.4f} ms   {bwd_time:>10.4f} ms   {total_time:>10.4f} ms")

            except Exception as e:
                print(f"{config_str:<20} {name:<30} ERROR: {str(e)}")

        print("-" * 120)


def test_correctness():
    """Test correctness and metrics"""
    batch_size, seq_len, vocab_size = 4, 8, 1000
    pad_idx = 0

    # Create sample data
    predict = torch.randn(batch_size, seq_len, vocab_size)
    target = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Add some padding
    target[:, -2:] = pad_idx

    print("\n" + "=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)

    losses = {
        'Original': OriginalCrossEntropyLoss(len_vocab=vocab_size, smoothing=0.1),
        'Optimized': OptimizedCrossEntropyLoss(vocab_size, pad_idx=pad_idx, label_smoothing=0.1),
        'Label Smoothing': LabelSmoothingLoss(vocab_size, pad_idx=pad_idx, smoothing=0.1),
        'Focal Loss': FocalLoss(vocab_size, pad_idx=pad_idx),
    }

    for name, loss_fn in losses.items():
        print(f"\n{name}:")

        if name == 'Optimized':
            loss, metrics = loss_fn(predict, target, return_metrics=True)
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Perplexity: {metrics['perplexity']:.4f}")
            print(f"  Num tokens: {metrics['num_tokens']}")
        else:
            loss = loss_fn(predict, target)
            print(f"  Loss: {loss.item():.6f}")


def test_advanced_features():
    """Test advanced features"""
    batch_size, seq_len, vocab_size = 4, 8, 1000
    pad_idx = 0

    print("\n" + "=" * 80)
    print("ADVANCED FEATURES TEST")
    print("=" * 80)

    # Create sample data
    predict = torch.randn(batch_size, seq_len, vocab_size)
    target = torch.randint(1, vocab_size, (batch_size, seq_len))
    target[:, -2:] = pad_idx

    # Test TranslationLoss
    print("\n1. Translation Loss (with length penalty):")
    src_lengths = torch.tensor([10, 12, 8, 15])
    tgt_lengths = torch.tensor([8, 10, 7, 13])

    loss_fn = TranslationLoss(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        label_smoothing=0.1,
        length_penalty=0.1
    )

    loss, metrics = loss_fn(
        predict, target,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths
    )

    print(f"  Total loss: {loss.item():.6f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # Test CompositeLoss with knowledge distillation
    print("\n2. Composite Loss (with knowledge distillation):")
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    loss_fn = CompositeLoss(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        label_smoothing=0.1,
        kl_weight=0.5
    )

    loss, metrics = loss_fn(
        predict, target,
        teacher_logits=teacher_logits
    )

    print(f"  Total loss: {loss.item():.6f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def compare_with_padding():
    """Compare behavior with/without padding"""
    batch_size, seq_len, vocab_size = 32, 64, 10000
    pad_idx = 0

    print("\n" + "=" * 80)
    print("PADDING COMPARISON")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data with varying amounts of padding
    padding_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    for pad_ratio in padding_ratios:
        predict = torch.randn(batch_size, seq_len, vocab_size, device=device)
        target = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

        # Add padding
        num_pad = int(seq_len * pad_ratio)
        if num_pad > 0:
            target[:, -num_pad:] = pad_idx

        # Test with and without ignoring padding
        loss_ignore = OptimizedCrossEntropyLoss(vocab_size, pad_idx=pad_idx)
        loss_no_ignore = OptimizedCrossEntropyLoss(vocab_size, pad_idx=-100)  # Don't ignore

        loss_val_ignore, metrics_ignore = loss_ignore(predict, target, return_metrics=True)
        loss_val_no_ignore, metrics_no_ignore = loss_no_ignore(predict, target, return_metrics=True)

        print(f"\nPadding ratio: {pad_ratio:.1%}")
        print(f"  With ignore padding:")
        print(f"    Loss: {loss_val_ignore.item():.6f}, Accuracy: {metrics_ignore['accuracy']:.4f}")
        print(f"  Without ignore padding:")
        print(f"    Loss: {loss_val_no_ignore.item():.6f}, Accuracy: {metrics_no_ignore['accuracy']:.4f}")


def profile_memory():
    """Profile memory usage"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return

    batch_size, seq_len, vocab_size = 32, 256, 30000

    print("\n" + "=" * 80)
    print("MEMORY PROFILING")
    print("=" * 80)

    device = torch.device('cuda')

    losses = {
        'Optimized': OptimizedCrossEntropyLoss(vocab_size),
        'Label Smoothing': LabelSmoothingLoss(vocab_size),
        'Focal Loss': FocalLoss(vocab_size),
    }

    for name, loss_fn in losses.items():
        torch.cuda.reset_peak_memory_stats()

        predict = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        loss_fn = loss_fn.to(device)

        # Forward
        loss = loss_fn(predict, target)

        # Backward
        loss.backward()

        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2

        print(f"{name:<25} Peak memory: {peak_memory:>10.2f} MB")

        del predict, target, loss
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Test correctness
    test_correctness()

    # Test advanced features
    test_advanced_features()

    # Compare with padding
    compare_with_padding()

    # Comprehensive benchmark
    run_comprehensive_benchmark()

    # Memory profiling
    if torch.cuda.is_available():
        profile_memory()
