from abc import ABC, abstractmethod
import torch


class PathingGenerator(ABC):
    """Base class for pathing generators."""
    @abstractmethod
    def generate(self, num_envs: int) -> torch.Tensor:
        raise NotImplementedError


class Small2(PathingGenerator):
    """Generates the Small2 map from https://arxiv.org/pdf/1711.00832.pdf"""
    pathing = torch.tensor(
         [[[[1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]]]],
        dtype=torch.uint8)

    def __init__(self, device: str):
        self.device = device

    def generate(self, num_envs: int) -> torch.Tensor:
        return self.pathing.to(self.device).repeat((num_envs, 1, 1, 1))


class Small3(PathingGenerator):
    """Generates the Small3 map from https://arxiv.org/pdf/1711.00832.pdf"""
    pathing = torch.tensor(
       [[[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
          [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
          [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
          [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
          [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
          [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]],
       dtype=torch.uint8)

    def __init__(self, device: str):
        self.device = device

    def generate(self, num_envs: int) -> torch.Tensor:
        return self.pathing.to(self.device).repeat((num_envs, 1, 1, 1))


class Small4(PathingGenerator):
    """Generates the Small4 map from https://arxiv.org/pdf/1711.00832.pdf"""
    pathing = torch.tensor(
        [[[[1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 1, 0, 1, 0, 0, 1],
           [1, 0, 1, 1, 0, 1, 1, 0, 1],
           [1, 0, 0, 1, 0, 1, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1]]]], dtype=torch.uint8)

    def __init__(self, device: str):
        self.device = device

    def generate(self, num_envs: int) -> torch.Tensor:
        return self.pathing.to(self.device).repeat((num_envs, 1, 1, 1))


class Random(PathingGenerator):
    """Generates a random pathing map"""
