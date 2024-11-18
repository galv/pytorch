import torch


def f(pred):
    return torch.cond(pred, torch.sin, torch.cos)
