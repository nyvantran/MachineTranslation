import torch

DEVICES = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 10
PIN_MEMORY = True
SMOOTHING = 0.1