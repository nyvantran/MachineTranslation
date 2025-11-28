import torch
import torch.nn as nn

# 1. Tạo một tầng Linear
# Mặc định PyTorch dùng Kaiming Uniform (cho Linear)
layer = nn.Linear(in_features=100, out_features=50)

print("Trọng số trước khi init (std mẫu):", layer.weight.std().item())

# 2. Áp dụng Xavier Normal
# gain=1.0 là mặc định cho Sigmoid/Tanh
nn.init.xavier_normal_(layer.weight, gain=1.0)

print("Trọng số sau khi init Xavier:", layer.weight.std().item())

# Kiểm tra lý thuyết:
# std lý thuyết = sqrt(2 / (100 + 50)) = sqrt(2/150) ≈ 0.115