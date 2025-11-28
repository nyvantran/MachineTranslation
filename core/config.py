import torch

DEVICES = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 10
PIN_MEMORY = True
SMOOTHING = 0.1
USE_AMP = False
ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0
MAX_LEN = 100
NUM_WORKERS = 4
