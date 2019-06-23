from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
import numpy as np
from PIL import Image

from wurm._filters import ORIENTATION_DELTAS

class VecEnv(ABC):
    @abstractmethod
    def step(self, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, dict):
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Optional[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def render(self, mode: str = 'human', env: Optional[int] = None) -> Any:
        raise NotImplementedError


class MultiagentVecEnv(ABC):
    @abstractmethod
    def step(self, actions: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict):
        raise NotImplementedError

    @abstractmethod
    def reset(self, done: torch.Tensor = None, return_observations: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def render(self, mode: str = 'human', env: Optional[int] = None) -> Any:
        raise NotImplementedError


def check_multi_vec_env_actions(actions: Dict[str, torch.Tensor], num_envs: int, num_agents: int):
    if len(actions) != num_agents:
        raise RuntimeError('Must have a Tensor of actions for each snake')

    for agent, act in actions.items():
        if act.dtype not in (torch.short, torch.int, torch.long):
            raise TypeError('actions Tensor must be an integer type i.e. '
                            '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

        if act.shape[0] != num_envs:
            raise RuntimeError('Must have the same number of actions as environments.')


def build_render_rgb(
        img: torch.Tensor,
        num_envs: int,
        env_size: int,
        num_rows: int,
        num_cols: int,
        render_size: int,
        env: Optional[int] = None) -> np.ndarray:
    """Util for viewing VecEnvs in a human friendly way.

    Args:
        img: Batch of RGB Tensors of the envs. Shape = (num_envs, 3, env_size, env_size).
        num_envs: Number of envs inside the VecEnv.
        env_size: Size of VecEnv.
        num_rows: Number of rows of envs to view.
        num_cols: Number of columns of envs to view.
        render_size: Pixel size of each viewed env.
        env: Optional specified environment to view.
    """
    # Convert to numpy
    img = img.cpu().numpy()

    # Rearrange images depending on number of envs
    if num_envs == 1 or env is not None:
        num_cols = num_rows = 1
        img = img[env or 0]
        img = np.transpose(img, (1, 2, 0))
    else:
        num_rows = num_rows
        num_cols = num_cols
        # Make a grid of images
        output = np.zeros((env_size * num_rows, env_size * num_cols, 3))
        for i in range(num_rows):
            for j in range(num_cols):
                output[
                i * env_size:(i + 1) * env_size, j * env_size:(j + 1) * env_size, :
                ] = np.transpose(img[i * num_cols + j], (1, 2, 0))

        img = output

    img = np.array(Image.fromarray(img.astype(np.uint8)).resize(
        (render_size * num_cols,
         render_size * num_rows)
    ))

    return img


def move_pixels(pixels: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Takes in pixels and directions, returns updated heads

    Args:
        pixels:
        directions:
    """
    # Create head position deltas
    filters = ORIENTATION_DELTAS.to(dtype=pixels.dtype, device=pixels.device)
    head_deltas = F.conv2d(
        pixels,
        filters,
        padding=1
    )
    directions_onehot = F.one_hot(directions, 4).to(pixels.dtype)
    head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, directions_onehot]).unsqueeze(1)
    pixels += head_deltas
    return pixels
