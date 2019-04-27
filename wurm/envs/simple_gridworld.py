from time import time
from collections import namedtuple
import torch
from torch.nn import functional as F
from typing import Tuple

from config import DEFAULT_DEVICE, BODY_CHANNEL, EPS, HEAD_CHANNEL, FOOD_CHANNEL
from wurm._filters import ORIENTATION_FILTERS, NO_CHANGE_FILTER
from wurm.utils import head, food, body, drop_duplicates


Spec = namedtuple('Spec', ['reward_threshold'])


class SimpleGridworld(object):
    """Batched gridworld environment.

    In this environment the agent can move in the 4 cardinal directions and receives +1 reward when moving on to a food
    square. At which point either the episode is finished or the food respawns. Moving off the edge of the gridworld
    results in a death.

    Each environment is represented as a Tensor image with 2 channels. Each channel has the following meaning:
    0 - Food channel. 1 = food, 0 = no food
    1 - Agent channel. 1 = agent location, 0 = empty

    Example of a single environment containing a single agent and a food object.

    Food channel                 Agent channel
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   | 1 |   |   |   |   |    |   |   |   |   | 1 |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    """

    spec = Spec(float('inf'))

    def __init__(self,
                 num_envs: int,
                 size: int,
                 on_death: str = 'restart',
                 observation_mode: str = 'default',
                 device: str = DEFAULT_DEVICE,
                 start_location: Tuple[int, int] = None,
                 manual_setup: bool = False,
                 verbose: int = 0):
        """Initialise the environments

        Args:
            num_envs:
            size:
            on_death:
        """
        self.num_envs = num_envs
        self.size = size
        self.on_death = on_death
        self.observation_mode = observation_mode
        self.start_location = start_location
        self.device = device
        self.verbose = verbose

        self.t = 0

        if manual_setup:
            # All zeros, user must create environment
            self.envs = torch.zeros((num_envs, 2, size, size)).to(self.device).requires_grad_(False)
        else:
            # Create environments automatically
            self.envs = self._create_envs(self.num_envs)
            self.envs.requires_grad_(False)

        self.done = torch.zeros(num_envs).to(self.device).byte()

        self.viewer = None

        self.head_colour = torch.Tensor((0, 255, 0)).short().to(self.device)
        self.food_colour = torch.Tensor((255, 0, 0)).short().to(self.device)
        self.edge_colour = torch.Tensor((0, 0, 0)).short().to(self.device)

    def _get_rgb(self):
        # RGB image same as is displayed in .render()
        img = torch.zeros((self.num_envs, 3, self.size, self.size)).short().to(self.device).requires_grad_(False) * 255

        # Convert to BHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))

        head_locations = (head(self.envs) > EPS).squeeze(1)
        img[head_locations, :] = self.head_colour

        food_locations = (food(self.envs) > EPS).squeeze(1)
        img[food_locations, :] = self.food_colour

        img[:, :1, :, :] = self.edge_colour
        img[:, :, :1, :] = self.edge_colour
        img[:, -1:, :, :] = self.edge_colour
        img[:, :, -1:, :] = self.edge_colour

        # Convert back to BCHW axes
        img = img.permute((0, 3, 1, 2))

        return img

    def _observe(self, observation_mode: str = 'default'):
        if observation_mode == 'default':
            # RGB image same as is displayed in .render()
            observation = self._get_rgb()

            # Normalise to 0-1
            observation = observation.float() / 255

            return observation
        elif observation_mode == 'raw':
            return self.envs.clone()
        elif observation_mode == 'positions':
            head_idx = self.envs[:, HEAD_CHANNEL, :, :].view(self.num_envs, self.size ** 2).argmax(dim=-1)
            food_idx = self.envs[:, FOOD_CHANNEL, :, :].view(self.num_envs, self.size ** 2).argmax(dim=-1)
            observation = torch.Tensor([
                head_idx // self.size,
                head_idx % self.size,
                food_idx // self.size,
                food_idx % self.size
            ]).float().unsqueeze(0)
            return observation
        else:
            raise Exception

    def step(self, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, dict):
        if actions.dtype not in (torch.short, torch.int, torch.long):
            raise TypeError('actions Tensor must be an integer type i.e. '
                            '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

        if actions.shape[0] != self.num_envs:
            raise RuntimeError('Must have the same number of actions as environments.')

        reward = torch.zeros((self.num_envs,)).float().to(self.device).requires_grad_(False)
        done = torch.zeros((self.num_envs,)).byte().to(self.device).requires_grad_(False)
        info = dict()

        t0 = time()
        # Create head position deltas
        head_deltas = F.conv2d(head(self.envs), ORIENTATION_FILTERS.to(self.device), padding=1)
        # Select the head position delta corresponding to the correct action
        actions_onehot = torch.FloatTensor(self.num_envs, 4).to(self.device)
        actions_onehot.zero_()
        actions_onehot.scatter_(1, actions.unsqueeze(-1), 1)
        head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, actions_onehot]).unsqueeze(1)

        # Move head position by applying delta
        self.envs[:, HEAD_CHANNEL:HEAD_CHANNEL + 1, :, :].add_(head_deltas).round_()
        if self.verbose:
            print(f'Head movement: {time() - t0}s')

        ################
        # Apply update #
        ################

        t0 = time()
        # Remove food and give reward
        # `food_removal` is 0 except where a snake head is at the same location as food where it is -1
        food_removal = head(self.envs) * food(self.envs) * -1
        reward.sub_(food_removal.view(self.num_envs, -1).sum(dim=-1).float())
        self.envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_removal
        if self.verbose:
            print(f'Food removal: {time() - t0}s')

        # Add new food if necessary.
        if food_removal.sum() < 0:
            t0 = time()
            food_addition_env_indices = (food_removal * -1).view(self.num_envs, -1).sum(dim=-1).byte()
            add_food_envs = self.envs[food_addition_env_indices, :, :, :]
            food_addition = self._get_food_addition(add_food_envs)
            self.envs[food_addition_env_indices, FOOD_CHANNEL:FOOD_CHANNEL+1, :, :] += food_addition
            if self.verbose:
                print(f'Food addition ({food_addition_env_indices.sum().item()} envs): {time() - t0}s')

        t0 = time()
        # Check for boundary, Done by performing a convolution with no padding
        # If the head is at the edge then it will be cut off and the sum of the head
        # channel will be 0
        edge_collision = F.conv2d(
            head(self.envs),
            NO_CHANGE_FILTER.to(self.device),
        ).view(self.num_envs, -1).sum(dim=-1) < EPS
        done = done | edge_collision
        info.update({'edge_collision': edge_collision})
        if self.verbose:
            print(f'Edge collision ({edge_collision.sum().item()} envs): {time() - t0}s')

        # Apply rounding to stop numerical errors accumulating
        self.envs.round_()

        self.done = done

        return self._observe(self.observation_mode), reward.unsqueeze(-1), done.unsqueeze(-1), info

    def _select_from_available_locations(self, locs: torch.Tensor) -> torch.Tensor:
        locations = torch.nonzero(locs)
        random_loc = locations[torch.randperm(locations.shape[0])[:1]]
        return random_loc

    def _get_food_addition(self, envs: torch.Tensor):
        # Get empty locations
        available_locations = envs.sum(dim=1, keepdim=True) < EPS
        # Remove boundaries
        available_locations[:, :, :1, :] = 0
        available_locations[:, :, :, :1] = 0
        available_locations[:, :, -1:, :] = 0
        available_locations[:, :, :, -1:] = 0

        food_indices = drop_duplicates(torch.nonzero(available_locations), 0)
        food_addition = torch.sparse_coo_tensor(
            food_indices.t(),  torch.ones(len(food_indices)), available_locations.shape, device=self.device)
        food_addition = food_addition.to_dense()

        return food_addition

    def reset(self, done: torch.Tensor = None):
        """Resets environments in which the snake has died

        Args:
            done: A 1D Tensor of length self.num_envs. A value of 1 means the corresponding environment needs to be
                reset
        """
        if done is None:
            done = self.done

        done = done.view((done.shape[0]))

        t0 = time()
        if done.sum() > 0:
            new_envs = self._create_envs(int(done.sum().item()))
            self.envs[done.byte(), :, :, :] = new_envs

        if self.verbose:
            print(f'Resetting {done.sum().item()} envs: {time() - t0}s')

        return self._observe(self.observation_mode)

    def _create_envs(self, num_envs: int):
        """Vectorised environment creation. Creates self.num_envs environments simultaneously."""
        if self.size <= 4:
            raise NotImplementedError('Environemnts smaller than this don\'t make sense.')

        envs = torch.zeros((num_envs, 2, self.size, self.size)).to(self.device)

        if self.start_location is None:
            available_locations = torch.zeros_like(envs)
            available_locations[:, :, :1, :] = 0
            available_locations[:, :, :, :1] = 0
            available_locations[:, :, -1:, :] = 0
            available_locations[:, :, :, -1:] = 0
            raise NotImplementedError("Haven't implemented random starting locations")
        else:
            envs[:, HEAD_CHANNEL:HEAD_CHANNEL + 1, self.start_location[0], self.start_location[1]] = 1

        # Add food
        food_addition = self._get_food_addition(envs)
        envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_addition

        return envs.round()

    def _consistent(self):
        pass
