import torch
from typing import Dict, Optional
import numpy as np
from gym.envs.classic_control import rendering

from .core import MultiagentVecEnv, check_multi_vec_env_actions, build_render_rgb
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
        self.x = torch.zeros((num_envs * num_agents), dtype=torch.long, device=self.device, requires_grad=False)
        self.y = torch.zeros((num_envs * num_agents), dtype=torch.long, device=self.device, requires_grad=False)
        self.hp = torch.ones((num_envs * num_agents), dtype=torch.long, device=self.device, requires_grad=False)

        # Environment outputs
        self.rewards = torch.zeros(self.num_envs * self.num_agents, dtype=torch.float, device=self.device)
        self.dones = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=self.device)

    def step(self, actions: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict):
        check_multi_vec_env_actions(actions, self.num_envs, self.num_agents)
        info = {}

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
        # Black background, pathing is grey
        img = torch.zeros((self.num_envs, 3, self.size, self.size), device=self.device, dtype=torch.short)
        img[self.pathing.expand_as(img)] += 127

        # Add colours for agents
        locations = (self.agents > EPS).squeeze().float()
        body_colours = torch.einsum('nhw,nc->nchw', [locations, self.agent_colours.float()])\
            .reshape(self.num_envs, self.num_agents, 3, self.size, self.size)\
            .sum(dim=1)\
            .short()
        print(img.shape, body_colours.shape)
        print(img.dtype, body_colours.dtype)
        print(img.device, body_colours.device)
        img += body_colours

        return img

    def _get_n_colours(self, n: int) -> torch.Tensor:
        colours = torch.rand((n, 3), device=self.device)
        colours[:, 0] /= 1.5  # Reduce red
        colours /= colours.norm(2, dim=1, keepdim=True)
        colours *= 192
        colours = colours.short()
        return colours

    def _observe(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


