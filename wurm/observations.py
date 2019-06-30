from abc import ABC, abstractmethod
from typing import Union, Dict
from collections import OrderedDict
import torch
import torch.nn.functional as F

from wurm.core import VecEnv, MultiagentVecEnv


class ObservationFunction(ABC):
    @abstractmethod
    def observe(self, env: MultiagentVecEnv, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class RenderObservations(ObservationFunction):
    def observe(self, env: MultiagentVecEnv, **kwargs) -> Dict[str, torch.Tensor]:
        img = env._get_env_images()
        img = img.repeat_interleave(env.num_agents, 0).float() / 255

        dict_observations = OrderedDict([
            (f'agent_{i}', obs) for i, obs in enumerate(
                img.view(env.num_envs, env.num_agents, 3, env.height, env.width).unbind(dim=1))
        ])

        return dict_observations


class FirstPersonCrop(ObservationFunction):
    def __init__(self, crop_filter: torch.Tensor):
        self.crop_filter = crop_filter

        self.h_diameter = int(crop_filter.cumsum(dim=2).max().item())
        self.h_radius = self.h_diameter // 2
        self.w_diameter = int(crop_filter.cumsum(dim=3).max().item())
        self.w_radius = self.w_diameter // 2

    def observe(self, env: MultiagentVecEnv, **kwargs) -> Dict[str, torch.Tensor]:
        img = env._get_env_images()

        # Normalise to 0-1
        img = img.float() / 255

        # Crop bits for each agent
        # Pad envs so we ge the correct size observation even when the head of the snake
        # is close to the edge of the environment
        padding = [self.h_radius, self.w_radius, ] * 2
        padded_img = F.pad(img, padding).repeat_interleave(env.num_agents, dim=0)

        n_living = (~env.dones).sum().item()
        padded_heads = F.pad(env.agents, padding)
        head_area_indices = F.conv2d(
            padded_heads,
            self.crop_filter,
            padding=(self.h_radius, self.w_radius)
        ).round()

        living_observations = padded_img[
            head_area_indices.expand_as(padded_img).byte()
        ]
        living_observations = living_observations.reshape(
            n_living, 3, self.h_diameter, self.w_diameter)

        observations = torch.zeros((env.num_envs * env.num_agents, 3, self.h_diameter, self.w_diameter),
                                   dtype=env.dtype, device=env.device)

        observations[~env.dones] = living_observations
        observations = observations \
            .reshape(env.num_envs, env.num_agents, 3, self.h_diameter, self.w_diameter)

        dict_observations = OrderedDict([
            (f'agent_{i}', observations[:, i].clone()) for i in range(env.num_agents)
        ])
        return dict_observations
