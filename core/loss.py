import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.0, len_vocab=None):
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.len_vocab = len_vocab if len_vocab is not None else 1000
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

    def forward(self, predict, target):
        # predict:[batch_size, seq_len, vocab_size]
        # target:[batch_size, seq_len]
        batch_size, seq_len, vocab_size = predict.size()
        predict = predict.view(-1, vocab_size)  # [batch_size*seq_len, vocab_size]
        target = target.view(-1)  # [batch_size*seq_len]
        loss = self.loss_fn(predict, target)
        return loss


def main():
    predictions = torch.tensor([
        [1.2, 0.5, -0.3, 2.5, 0.1],  # Mẫu 1
        [-0.5, 3.1, 0.0, 1.1, 1.9],  # Mẫu 2
        [0.1, 0.2, 0.3, 0.4, 4.5]  # Mẫu 3
    ], dtype=torch.float32)
    labels = torch.tensor([3, 1, 4], dtype=torch.long)
    loss_function = CrossEntropyLoss(smoothing=0.1, len_vocab=5)
    loss = loss_function(predictions.unsqueeze(1), labels.unsqueeze(1))
    print(f"Giá trị Loss: {loss.item()}")


if __name__ == "__main__":
    main()
