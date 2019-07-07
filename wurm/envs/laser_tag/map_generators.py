from abc import ABC, abstractmethod
from typing import List, NamedTuple
import torch


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


class LaserTagMap(NamedTuple):
    pathing: torch.Tensor
    respawn: torch.Tensor


class LaserTagMapGenerator(ABC):
    """Base class for map generators."""
    @abstractmethod
    def generate(self, num_envs: int) -> LaserTagMap:
        raise NotImplementedError


class FixedMapGenerator(LaserTagMapGenerator):
    _pathing: torch.Tensor
    _respawn: torch.Tensor

    def __init__(self, device: str):
        self.device = device

    def generate(self, num_envs: int) -> LaserTagMap:
        pathing = self._pathing.to(self.device).repeat((num_envs, 1, 1, 1))
        respawn = self._respawn.to(self.device).repeat((num_envs, 1, 1, 1))
        return LaserTagMap(pathing, respawn)

    @property
    def height(self):
        return self._pathing.size(2)

    @property
    def width(self):
        return self._pathing.size(3)


class MapFromString(FixedMapGenerator):
    def __init__(self, mapstring: List[str], device: str):
        super(MapFromString, self).__init__(device)
        self._pathing, self._respawn = parse_mapstring(mapstring)


class MapPool(LaserTagMapGenerator):
    """Uniformly selects maps at random from a pool of fixed maps."""
    def __init__(self, map_pool: List[FixedMapGenerator]):
        assert len(map_pool) > 0
        self.map_pool = map_pool
        self.device = self.map_pool[0].device
        self.height = self.map_pool[0].height
        self.width = self.map_pool[0].width

    def generate(self, num_envs: int) -> LaserTagMap:
        map_selection = torch.randint(0, len(self.map_pool), size=(num_envs, ))
        print(map_selection)

        pathing = torch.zeros((num_envs, 1, self.height, self.width), dtype=torch.uint8, device=self.device)
        respawn = torch.zeros((num_envs, 1, self.height, self.width), dtype=torch.uint8, device=self.device)

        for i in range(len(self.map_pool)):
            map_i = map_selection == i
            num_map_i = map_i.sum().item()
            new_maps = self.map_pool[i].generate(num_map_i)
            pathing[map_i] = new_maps.pathing
            respawn[map_i] = new_maps.respawn

        return LaserTagMap(pathing, respawn)


class Random(LaserTagMapGenerator):
    """Generates a random pathing map"""
