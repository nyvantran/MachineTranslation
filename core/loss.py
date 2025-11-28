import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CrossEntropyLoss(nn.Module):
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
        super(CrossEntropyLoss, self).__init__()

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
            return_metrics: khi mà True thì trả về thêm metrics
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

