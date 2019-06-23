from abc import ABC, abstractmethod
import torch
from typing import Optional, Dict, Tuple


class VecEnv(ABC):
    def step(self, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, dict):
        raise NotImplementedError

    def reset(self) -> Optional[torch.Tensor]:
        raise NotImplementedError


class MultiagentVecEnv(ABC):
    def step(self, actions: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict):
        raise NotImplementedError

    def reset(self, done: torch.Tensor = None, return_observations: bool = True) ->  Optional[Dict[str, torch.Tensor]]:
        raise NotImplementedError
