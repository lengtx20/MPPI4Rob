import torch

def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
