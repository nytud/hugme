import os
import torch


NUM_WORKERS = os.cpu_count()
NUM_GPUS = torch.cuda.device_count()
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICES = [f"cuda:{i}" for i in range(NUM_GPUS)] if NUM_GPUS else ["cpu"]


MODEL_CONFIG = {
    "gemma-2-2b-it": {
    },
}