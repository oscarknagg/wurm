"""This file contains some predetermined convolutional filters that are useful elsewhere."""
import torch

from config import DEFAULT_DEVICE


ORIENTATION_DELTAS = torch.tensor([
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


ORIENTATION_FILTERS = torch.tensor([
    [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ],
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ],
]).unsqueeze(1).float()


NO_CHANGE_FILTER = torch.tensor([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]).float().unsqueeze(0).unsqueeze(0)


LENGTH_3_SNAKES = torch.tensor([
    [
        [0, 1, 0],
        [0, 2, 0],
        [0, 3, 0],
    ],
    [
        [0, 0, 0],
        [3, 2, 1],
        [0, 0, 0],
    ],
    [
        [0, 3, 0],
        [0, 2, 0],
        [0, 1, 0],
    ],
    [
        [0, 0, 0],
        [1, 2, 3],
        [0, 0, 0],
    ],
]).unsqueeze(1).float()


# Doesn't quite work as expected
LENGTH_4_SNAKES = torch.tensor([
    [
        [0, 0, 0, 0, 0],
        [0, 0, 4, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [4, 3, 2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 4, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
]).unsqueeze(1).float()
