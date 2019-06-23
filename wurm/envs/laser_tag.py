import torch
from typing import Dict, Optional
import numpy as np
from gym.envs.classic_control import rendering
import torch.nn.functional as F

from wurm._filters import ORIENTATION_FILTERS
from .core import MultiagentVecEnv, check_multi_vec_env_actions, build_render_rgb, move_pixels
from config import DEFAULT_DEVICE, EPS


class LaserTag(MultiagentVecEnv):
    """Laser tag environment.

    This environment is meant to be a slightly extended version of the laser tag multiagent
    environment from Deepmind's paper: https://arxiv.org/pdf/1711.00832.pdf

    Actions:
        0: No-op
        1: Rotate right
        2: Rotate left
        3: Move forward
        4: Move back
        5: Move right
        6: Move left
        7: Fire laser
    """
    no_op = 0
    rotate_right = 1
    rotate_left = 2
    move_forward = 3
    move_back = 4
    move_right = 5
    move_left = 6
    fire = 7

    def __init__(self,
                 num_envs: int,
                 num_agents: int,
                 size: int,
                 render_args: dict = None,
                 dtype: torch.dtype = torch.float,
                 device: str = DEFAULT_DEVICE):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.size = size
        self.dtype = dtype
        self.device = device

        self.viewer = None
        self.agent_colours = self._get_n_colours(num_envs*num_agents)
        if render_args is None:
            self.render_args = {'num_rows': 1, 'num_cols': 1, 'size': 256}
        else:
            self.render_args = render_args

        # Environment tensors
        self.agents = torch.zeros((num_envs*num_agents, 1, size, size), dtype=self.dtype, device=self.device,
                                  requires_grad=False)
        self.pathing = torch.zeros((num_envs, 1, size, size), dtype=torch.uint8, device=self.device,
                                   requires_grad=False)
        self.pathing[:, :, :1, :] = 1
        self.pathing[:, :, :, :1] = 1
        self.pathing[:, :, -1:, :] = 1
        self.pathing[:, :, :, -1:] = 1
        self.orientations = torch.zeros((num_envs * num_agents), dtype=torch.long, device=self.device,
                                        requires_grad=False)
        # self.x = torch.zeros((num_envs * num_agents), dtype=torch.long, device=self.device, requires_grad=False)
        # self.y = torch.zeros((num_envs * num_agents), dtype=torch.long, device=self.device, requires_grad=False)
        self.hp = torch.ones((num_envs * num_agents), dtype=torch.long, device=self.device, requires_grad=False)

        # Environment outputs
        self.rewards = torch.zeros(self.num_envs * self.num_agents, dtype=torch.float, device=self.device)
        self.dones = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=self.device)

    def step(self, actions: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict):
        check_multi_vec_env_actions(actions, self.num_envs, self.num_agents)
        info = {}
        actions = torch.stack([v for k, v in actions.items()]).t().flatten()

        # Update orientations
        self.orientations[actions == self.rotate_right] += 1
        self.orientations[actions == self.rotate_left] += 3
        self.orientations.fmod_(4)

        # Movement
        has_moved = actions == self.move_forward
        has_moved |= actions == self.move_back
        has_moved |= actions == self.move_right
        has_moved |= actions == self.move_left
        if torch.any(has_moved):
            # Default movement is in the orientation direction
            directions = self.orientations.clone()
            # Backwards is reverse of the orientation direction
            directions[actions == self.move_back] = directions[actions == self.move_back] + 2
            # Right is orientation +1
            directions[actions == self.move_right] = directions[actions == self.move_right] + 1
            # Left is orientation -1
            directions[actions == self.move_left] = directions[actions == self.move_left] + 3
            # Keep in range(4)
            directions.fmod_(4)
            self.agents[has_moved] = move_pixels(self.agents[has_moved], directions[has_moved])

        observations = self._observe()

        return observations, self.rewards, self.dones, info

    def reset(self, done: torch.Tensor = None, return_observations: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        pass

    def render(self, mode: str = 'human', env: Optional[int] = None):
        if self.viewer is None and mode == 'human':
            self.viewer = rendering.SimpleImageViewer()

        img = self._get_env_images()
        img = build_render_rgb(img=img, num_envs=self.num_envs, env_size=self.size, env=env,
                               num_rows=self.render_args['num_rows'], num_cols=self.render_args['num_cols'],
                               render_size=self.render_args['size'])

        if mode == 'human':
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return img
        else:
            raise ValueError('Render mode not recognised.')

    def _get_env_images(self) -> torch.Tensor:
        # Black background
        img = torch.zeros((self.num_envs, 3, self.size, self.size), device=self.device, dtype=torch.short)

        # Add slight highlight for orientation
        locations = (self.agents > EPS).float()
        filters = ORIENTATION_FILTERS.to(dtype=self.dtype, device=self.device)
        orientation_highlights = F.conv2d(
            locations,
            filters,
            padding=1
        ) * 31
        directions_onehot = F.one_hot(self.orientations, 4).to(self.dtype)
        orientation_highlights = torch.einsum('bchw,bc->bhw', [orientation_highlights, directions_onehot])\
            .view(self.num_envs, self.num_agents, 1, self.size, self.size)\
            .sum(dim=1)\
            .short()
        img += orientation_highlights.expand_as(img)

        # Add colours for agents
        body_colours = torch.einsum('nhw,nc->nchw', [locations.squeeze(), self.agent_colours.float()])\
            .view(self.num_envs, self.num_agents, 3, self.size, self.size)\
            .sum(dim=1)\
            .short()
        img += body_colours

        # Walls are grey
        img[self.pathing.expand_as(img)] = 127

        return img

    def _get_n_colours(self, n: int) -> torch.Tensor:
        colours = torch.rand((n, 3), device=self.device)
        colours /= colours.norm(2, dim=1, keepdim=True)
        colours *= 192
        colours = colours.short()
        return colours

    def _observe(self) -> Dict[str, torch.Tensor]:
        return {f'agent_{i}': torch.zeros((self.num_envs, 1)) for i in range(self.num_agents)}

    @property
    def x(self):
        return self.agents.view(self.num_envs*self.num_agents, -1).argmax(dim=1) // self.size

    @property
    def y(self):
        return self.agents.view(self.num_envs*self.num_agents, -1).argmax(dim=1) % self.size
