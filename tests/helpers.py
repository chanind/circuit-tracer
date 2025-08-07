import torch
from torch import device

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = device(DEVICE_STR)
