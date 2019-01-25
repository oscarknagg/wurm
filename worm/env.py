import torch


class World(object):
    def __init__(self, size: int):
        self.size = size

        # Channels:
        # 0: Food
        # 1: Snake
        self.world = torch.zeros((1, 2, size, size))

