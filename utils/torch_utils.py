import torch
import numpy as np


def load_tensor_from_npy(path):
    return torch.from_numpy(np.load(path)).float()