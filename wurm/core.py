from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

import numpy as np
import torch
from PIL import Image
from gym.envs.classic_control import rendering
from torch.nn import functional as F

from wurm._filters import ORIENTATION_DELTAS


class VecEnv(ABC):
    def __init__(self, num_envs: int, num_agents: int, height: int, width: int):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.height = height
        self.width = width

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
    def __init__(self, num_envs: int,
                 num_agents: int,
                 height: int,
                 width: int,
                 dtype: torch.dtype,
                 device: str):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.height = height
        self.width = width
        self.dtype = dtype
        self.device = device
        self.viewer = None

        # This Tensor represents the location of each agent in each environment. It should contain
        # only one non-zero entry for each sub array along dimension 0.
        self.agents = torch.zeros((num_envs * num_agents, 1, height, width), dtype=dtype, device=device, requires_grad=False)

        # This Tensor represents the current alive/dead state of each agent in each environment
        self.dones = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=device, requires_grad=False)

    @abstractmethod
    def step(self, actions: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict):
        raise NotImplementedError

    @abstractmethod
    def reset(self, done: torch.Tensor = None, return_observations: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def _get_env_images(self) -> torch.Tensor:
        """Gets RGB arrays for each environment.

        Returns:
            img: A Tensor of shape (num_envs, 3, height, width) and dtype torch.short i.e.
                an RBG rendering of each environment
        """
        raise NotImplementedError

    def render(self, mode: str = 'human', env: Optional[int] = None) -> Any:
        if self.viewer is None and mode == 'human':
            self.viewer = rendering.SimpleImageViewer()

        img = self._get_env_images()
        img = build_render_rgb(img=img, num_envs=self.num_envs, env_height=self.height, env_width=self.width, env=env,
                               num_rows=self.render_args['num_rows'], num_cols=self.render_args['num_cols'],
                               render_size=self.render_args['size'])

        if mode == 'human':
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return img
        else:
            raise ValueError('Render mode not recognised.')


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
        env_height: int,
        env_width: int,
        num_rows: int,
        num_cols: int,
        render_size: int,
        env: Optional[int] = None) -> np.ndarray:
    """Util for viewing VecEnvs in a human friendly way.

    Args:
        img: Batch of RGB Tensors of the envs. Shape = (num_envs, 3, env_size, env_size).
        num_envs: Number of envs inside the VecEnv.
        env_height: Size of VecEnv.
        env_width: Size of VecEnv.
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
        output = np.zeros((env_height * num_rows, env_width * num_cols, 3))
        for i in range(num_rows):
            for j in range(num_cols):
                output[
                i * env_height:(i + 1) * env_height, j * env_width:(j + 1) * env_width, :
                ] = np.transpose(img[i * num_cols + j], (1, 2, 0))

        img = output

    ratio = env_width / env_height

    img = np.array(Image.fromarray(img.astype(np.uint8)).resize(
        (int(render_size * num_cols * ratio),
         int(render_size * num_rows))
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