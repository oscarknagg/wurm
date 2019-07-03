from abc import ABC, abstractmethod
from typing import Union, Dict
from collections import OrderedDict
import torch
import torch.nn.functional as F

from wurm.core import VecEnv, MultiagentVecEnv
from .utils import rotate_image_batch, pad_to_square


class ObservationFunction(ABC):
    """Base class for observation functions."""
    @abstractmethod
    def observe(self, env: MultiagentVecEnv, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class RenderObservations(ObservationFunction):
    """Shows the human readable render to each agent.

    NB:
    - This makes environments fully observable.
    - This observation type can make it hard for agents to identify themselves in the multiagent setting.
    """
    def observe(self, env: MultiagentVecEnv, **kwargs) -> Dict[str, torch.Tensor]:
        img = env._get_env_images()
        img = img.repeat_interleave(env.num_agents, 0).float() / 255

        dict_observations = OrderedDict([
            (f'agent_{i}', obs) for i, obs in enumerate(
                img.view(env.num_envs, env.num_agents, 3, env.height, env.width).unbind(dim=1))
        ])

        return dict_observations


class FirstPersonCrop(ObservationFunction):
    def __init__(self,
                 height: int = None,
                 width: int = None,
                 first_person_rotation: bool = False,
                 in_front: int = None,
                 behind: int = None,
                 side: int = None,
                 padding_value: int = 0):
        """Crops the full observation in an area around each living agent.

        If `first_person_rotation` is False then the observations are a simple oblong crop around agent locations.

        If `first_person_rotation` is True then the crops are rotated so that the agent always to be facing the same
        direction in its own observations.

        Args:
            height: Height of observation.
            width: Width of observation.
            first_person_rotation: Whether or not to apply first person rotation.
            in_front:
            behind:
            side:
            padding_value:
        """
        self.padding_value = padding_value

        # Check arguments
        if first_person_rotation:
            if not (height is None and width is None):
                raise ValueError('Please specify {in_front, behind, side} instead of {height, width}.')
        else:
            if not (in_front is None and behind is None and side is None):
                raise ValueError('Please specify {height, width} instead of {in_front, behind, side}.')

        # Build the cropping filter
        if first_person_rotation:
            h = 2 * in_front + 1
            w = 2 * side + 1
            self.crop_filter = torch.ones((1, 1, h, w))
            self.crop_filter[:, :, :(h // 2) - behind] = 0
        else:
            self.crop_filter = torch.ones((1, 1, height, width))

        self.first_person_rotation = first_person_rotation

        self.h_diameter = int(self.crop_filter.cumsum(dim=2).max().item())
        self.h_radius = self.h_diameter // 2
        self.w_diameter = int(self.crop_filter.cumsum(dim=3).max().item())
        self.w_radius = self.w_diameter // 2

        if self.first_person_rotation:
            self.padding = [self.crop_filter.size(2) // 2, self.crop_filter.size(3) // 2]
        else:
            self.padding = [self.h_radius, self.w_radius, ]

        self.first_observation = True

    def _handle_first_observation(self, env: MultiagentVecEnv):
        if self.first_person_rotation and not hasattr(env, 'orientations'):
            raise ValueError('Environment must track orientations to enable rotated first person cropping.')

        self.crop_filter = self.crop_filter.to(device=env.device)

        self.first_observation = False

    def observe(self, env: MultiagentVecEnv, **kwargs) -> Dict[str, torch.Tensor]:
        if self.first_observation:
            self._handle_first_observation(env)

        img = env._get_env_images()
        agents = env.agents.clone()

        # Pad to square if the h != w because we can only use rotate_image_batch on
        # square images
        img, agents = pad_to_square(img, self.padding_value), pad_to_square(agents)

        # Normalise to 0-1
        img = img.float() / 255
        img = img.repeat_interleave(env.num_agents, dim=0)

        if self.first_person_rotation:
            # Orientation pre-processing
            orientation_preprocessing = [
                (0, 180, 180),
                (1, 90, 270),
                (2, 0, 0),
                (3, 270, 90),
            ]
            for orientation, rotation, _ in orientation_preprocessing:
                _orientation = env.orientations == orientation
                if torch.any(_orientation):
                    img[_orientation] = rotate_image_batch(img[_orientation], degree=rotation)
                    agents[_orientation] = rotate_image_batch(agents[_orientation], degree=rotation)

        # Crop bits for each agent
        # Pad envs so we ge the correct size observation even when agents
        # are close to the edge of the environment
        padding = self.padding * 2
        padded_img = F.pad(img, padding, value=self.padding_value / 255)
        n_living = (~env.dones).sum().item()
        padded_agents = F.pad(agents, padding)
        agent_area_mask = F.conv2d(
            padded_agents,
            self.crop_filter,
            padding=self.padding,
        ).round()

        living_observations = padded_img[
            agent_area_mask.expand_as(padded_img).byte()
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
