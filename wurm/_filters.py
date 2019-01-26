"""This file contains some predetermined convolutional filters that are useful elsewhere."""
import torch


ORIENTATION_FILTERS = torch.Tensor([
    [
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
    ],
    [
        [0, 0, 0],
        [1, -1, 0],
        [0, 0, 0],
    ],
]).unsqueeze(1).float()


NO_CHANGE_FILTER = torch.Tensor([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]).float().unsqueeze(0).unsqueeze(0)
