import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_sequences(features, seq_len):

    X, y = [], []

    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(features[i+seq_len, 0])

    return np.array(X), np.array(y)

def get_device():
    """
    Get available device.
    """

    return torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )    