from abc import ABC, abstractmethod
from typing import List
import torch

from wurm.envs.laser_tag import maps


def parse_mapstring(mapstring: List[str]) -> (torch.Tensor, torch.Tensor):
    # Get height and width
    height = len(mapstring)
    width = (len(mapstring[0]) + 1) // 2

    # Check consistent height and width
    # Convert to tensor
    pathing = torch.zeros((1, 1, height, width), dtype=torch.uint8)
    respawn = torch.zeros((1, 1, height, width), dtype=torch.uint8)
    for i, line in enumerate(mapstring):
        # Remove padding spaces
        line = (line + ' ')[::2]

        if len(line) != width:
            raise ValueError('Map string has inconsistent shape')

        _pathing = torch.tensor([char == '*' for char in line])
        pathing[:, :, i, :] = _pathing
        _respawn = torch.tensor([char == 'P' for char in line])
        respawn[:, :, i, :] = _respawn

    return pathing, respawn


class LaserTagMapGenerator(ABC):
    """Base class for pathing generators."""
    @abstractmethod
    def pathing(self, num_envs: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def respawns(self, num_envs: int) -> torch.Tensor:
        raise NotImplementedError


class FixedMapGenerator(LaserTagMapGenerator):
    _pathing: torch.Tensor
    _respawn: torch.Tensor

    def __init__(self, device: str):
        self.device = device

    def pathing(self, num_envs: int) -> torch.Tensor:
        return self._pathing.to(self.device).repeat((num_envs, 1, 1, 1))

    def respawns(self, num_envs: int) -> torch.Tensor:
        return self._respawn.to(self.device).repeat((num_envs, 1, 1, 1))


class MapFromString(FixedMapGenerator):
    def __init__(self, mapstring: List[str], device: str):
        super(MapFromString, self).__init__(device)
        self._pathing, self._respawn = parse_mapstring(mapstring)


class MapPool(LaserTagMapGenerator):
    """Uniformly selects maps at random from a pool of fixed maps."""
    def __init__(self, map_pool: List[FixedMapGenerator]):
        self.map_pool = map_pool


# class Small2(FixedMapGenerator):
#     """Generates the Small2 map from https://arxiv.org/pdf/1711.00832.pdf"""
#     def __init__(self, device: str):
#         return MapFromString(device=device, mapstring=_maps.small2)


# def Small2(device: str):
#     return MapFromString(device=device, mapstring=_maps.small2)

    # _pathing = torch.tensor(
    #      [[[[1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [1, 0, 1, 1, 0, 1, 1, 0, 1],
    #         [1, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1]]]], dtype=torch.uint8)
    # _respawn = torch.tensor(
    #    [[[[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #       [0, 1, 0, 0, 0, 0, 0, 1, 0],
    #       [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #       [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #       [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #       [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #       [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #       [0, 1, 0, 0, 0, 0, 0, 1, 0],
    #       [0, 0, 0, 0, 0, 0, 0, 0, 0]]]], dtype=torch.uint8)
#
#
# class Small3(FixedMapGenerator):
#     """Generates the Small3 map from https://arxiv.org/pdf/1711.00832.pdf"""
#     _pathing = torch.tensor(
#        [[[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
#           [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#           [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]], dtype=torch.uint8)
#     _respawn = torch.tensor(
#        [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]], dtype=torch.uint8)
#
#
# class Small4(FixedMapGenerator):
#     """Generates the Small4 map from https://arxiv.org/pdf/1711.00832.pdf"""
#     _pathing = torch.tensor(
#        [[[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
#           [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]], dtype=torch.uint8)
#     _respawn = torch.tensor(
#        [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]], dtype=torch.uint8)


class Random(LaserTagMapGenerator):
    """Generates a random pathing map"""
