import torch
import numpy as np


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    return device
