from time import time, sleep
from typing import Dict, Tuple
from collections import namedtuple, OrderedDict
import torch
from torch.nn import functional as F
from gym.envs.classic_control import rendering
import numpy as np
from PIL import Image

from config import DEFAULT_DEVICE, BODY_CHANNEL, EPS, HEAD_CHANNEL, FOOD_CHANNEL
from wurm._filters import ORIENTATION_FILTERS, NO_CHANGE_FILTER, LENGTH_3_SNAKES
from wurm.utils import determine_orientations, drop_duplicates, env_consistency, snake_consistency, head, body, food


Spec = namedtuple('Spec', ['reward_threshold'])


class MultiSnake(object):
    """Batched snake environment.

    The dynamics of this environment aim to emulate that of the mobile phone game "Snake".

    Each environment is represented as a Tensor image with 3 channels. Each channel has the following meaning:
    0 - Food channel. 1 = food, 0 = no food
    1 - Head channel. 1 = head of snake, 0 = empty
    2 - Body channel. Each segment of the snake is represented as a positive integer. 1 is the tail of the
        snake and the maximum number is at the same position as the head of the snake in the head channel.

    Example of a single environment containing a snake of length 8, all unfilled squares represent 0s.

    Food channel                 Head channel                 Body channel
    +---+---+---+---+---+---+    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |    |   |   | 3 | 4 | 5 |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |    |   | 1 | 2 |   | 6 |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |    |   |   |   |   | 7 |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   | 1 |   |   |   |   |    |   |   |   |   | 1 |   |    |   |   |   |   | 8 |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+    +---+---+---+---+---+---+
    |   |   |   |   |   |   |    |   |   |   |   |   |   |    |   |   |   |   |   |   |
    +---+---+---+---+---+---+    +---+---+---+---+---+---+    +---+---+---+---+---+---+

    The advantage of this representation is that the dynamics of multiple environments can be stepped in a parallel
    fashion using using just tensor operations allowing one to run 1000s of envs in parallel on a single machine.
    """

    spec = Spec(float('inf'))
    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second': 12
    }

    def __init__(self,
                 num_envs: int,
                 num_snakes: int,
                 size: int,
                 initial_snake_length: int = 3,
                 on_death: str = 'restart',
                 observation_mode: str = 'full',
                 device: str = DEFAULT_DEVICE,
                 dtype: torch.dtype = torch.float,
                 manual_setup: bool = False,
                 food_on_death_prob: float = 0.5,
                 boost: bool = True,
                 boost_cost_prob: float = 0.5,
                 food_mode: str = 'only_one',
                 food_rate: float = 5e-4,
                 respawn_mode: str = 'all',
                 verbose: int = 0,
                 render_args: dict = None):
        """Initialise the environments

        Args:
            num_envs:
            size:
        """
        self.num_envs = num_envs
        self.num_snakes = num_snakes
        self.size = size
        self.initial_snake_length = initial_snake_length
        self.on_death = on_death
        self.device = device
        self.verbose = verbose
        self.dtype = dtype
        self.observation_mode = observation_mode
        if observation_mode.startswith('partial_'):
            self.observation_width = int(observation_mode.split('_')[1])
            self.observation_size = 2*int(observation_mode.split('_')[1]) + 1

        if render_args is None:
            self.render_args = {'num_rows': 1, 'num_cols': 1, 'size': 256}
        else:
            self.render_args = render_args

        self.foods = torch.zeros((num_envs, 1, size, size), dtype=self.dtype, device=self.device).requires_grad_(False)
        self.heads = torch.zeros((num_envs*num_snakes, 1, size, size), dtype=self.dtype, device=self.device).requires_grad_(False)
        self.bodies = torch.zeros((num_envs*num_snakes, 1, size, size), dtype=self.dtype, device=self.device).requires_grad_(False)
        self.dones = torch.zeros(self.num_envs * self.num_snakes, dtype=torch.uint8, device=self.device)
        self.rewards = torch.zeros(self.num_envs * self.num_snakes, dtype=torch.float, device=self.device)
        self.env_lifetimes = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.snake_lifetimes = torch.zeros((num_envs, num_snakes), dtype=torch.long, device=device)
        self.orientations = torch.zeros((num_envs*num_snakes), dtype=torch.long, device=self.device, requires_grad=False)
        self.viewer = 0
        self.boost_this_step = OrderedDict([
            (
                f'agent_{i}',
                torch.zeros((self.num_envs,), dtype=torch.uint8, device=self.device).requires_grad_(False)
            ) for i in range(self.num_snakes)
        ])

        if not manual_setup:
            # Create environments
            self.envs, self.orientations = self._create_envs(self.num_envs)
            self.envs.requires_grad_(False)

        ###################################
        # Environment dynamics parameters #
        ###################################
        self.respawn_mode = respawn_mode
        self.food_on_death_prob = food_on_death_prob
        self.boost = boost
        self.boost_cost_prob = boost_cost_prob
        self.food_mode = food_mode
        self.food_rate = food_rate
        self.max_food = 10
        self.max_env_lifetime = 5000

        # Rendering parameters
        self.viewer = None
        # Own snake appears green
        self.self_colour = torch.tensor((0, 192, 0), dtype=torch.short, device=self.device)
        self.self_boost_colour = torch.tensor((0, 255, 0), dtype=torch.short, device=self.device)
        # Other snakes appear blue
        self.other_colour = torch.tensor((0, 0, 192), dtype=torch.short, device=self.device)
        self.other_boost_colour = torch.tensor((0, 0, 255), dtype=torch.short, device=self.device)
        # Boost has different head colour
        self.food_colour = torch.tensor((255, 0, 0), dtype=torch.short, device=self.device)
        self.edge_colour = torch.tensor((0, 0, 0), dtype=torch.short, device=self.device)

        self.agent_colours = torch.tensor([
            [13, 92, 167],  # Blue ish
            [86, 163, 49],  # Green
            [133, 83, 109],  # Red-ish
            [135, 135, 4],   # Yellow ish
        ], device=self.device, dtype=torch.short)
        self.num_colours = self.agent_colours.shape[0]

        self.info = {}

        self.edge_locations_mask = torch.zeros(
            (1, 1, self.size, self.size), dtype=self.dtype, device=self.device)

        self.edge_locations_mask[:, :, :1, :] = 1
        self.edge_locations_mask[:, :, :, :1] = 1
        self.edge_locations_mask[:, :, -1:, :] = 1
        self.edge_locations_mask[:, :, :, -1:] = 1

    def get_subenv(self, i: int) -> torch.Tensor:
        return torch.cat([self.foods, self.heads[i:i+1], self.bodies[i:i+1]], dim=1)

    def _log(self, msg: str):
        if self.verbose > 0:
            print(msg)

    def _make_generic_rgb(self, colour_layers: Dict[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        img = torch.ones((self.num_envs, 3, self.size, self.size), dtype=torch.short, device=self.device) * 255

        # Convert to BHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))

        for locations, colour in colour_layers.items():
            img[locations, :] = colour

        img[:, :1, :, :] = self.edge_colour
        img[:, :, :1, :] = self.edge_colour
        img[:, -1:, :, :] = self.edge_colour
        img[:, :, -1:, :] = self.edge_colour

        # Convert back to BCHW axes
        img = img.permute((0, 3, 1, 2))

        return img

    def _get_env_images(self):
        # Regular snakes
        layers = {
            self.foods.gt(EPS).squeeze(1): self.food_colour,
        }

        layers.update({
            self.bodies.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1).gt(EPS): self.agent_colours[0] / 2,
            self.heads.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1).gt(EPS): self.agent_colours[0]
        })

        # for i, (agent, has_boosted) in enumerate(self.boost_this_step.items()):
        #     if has_boosted.sum() > 0:
        #         boosted_bodies = torch.zeros((self.num_envs, self.size, self.size), dtype=torch.uint8,
        #                                      device=self.device)
        #         boosted_heads = torch.zeros((self.num_envs, self.size, self.size), dtype=torch.uint8,
        #                                     device=self.device)
        #
        #         boosted_bodies[has_boosted] = self._bodies[has_boosted, i:i + 1].sum(dim=1).gt(EPS)
        #         boosted_heads[has_boosted] = self._heads[has_boosted, i:i + 1].sum(dim=1).gt(EPS)
        #         layers.update({
        #             boosted_bodies: (self.agent_colours[i % self.num_colours].float() * (2 / 3)).short(),
        #             boosted_heads: (self.agent_colours[i % self.num_colours].float() * (4 / 3)).short(),
        #         })
        #
        #     regular_bodies = torch.zeros((self.num_envs, self.size, self.size), dtype=torch.uint8, device=self.device)
        #     regular_heads = torch.zeros((self.num_envs, self.size, self.size), dtype=torch.uint8, device=self.device)
        #     regular_bodies[~has_boosted] = self._bodies[~has_boosted, i:i + 1].sum(dim=1).gt(EPS)
        #     regular_heads[~has_boosted] = self._heads[~has_boosted, i:i + 1].sum(dim=1).gt(EPS)
        #     layers.update({
        #         regular_bodies: self.agent_colours[i % self.num_colours] / 2,
        #         regular_heads: self.agent_colours[i % self.num_colours],
        #     })

        # Get RBG Tensor NCHW
        img = self._make_generic_rgb(layers)

        return img

    def render(self, mode: str = 'human'):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        img = self._get_env_images()

        # Convert to numpy
        img = img.cpu().numpy()

        # Rearrange images depending on number of envs
        if self.num_envs == 1:
            num_cols = num_rows = 1
            img = img[0]
            img = np.transpose(img, (1, 2, 0))
        else:
            num_rows = self.render_args['num_rows']
            num_cols = self.render_args['num_cols']
            # Make a 2x2 grid of images
            output = np.zeros((self.size * num_rows, self.size * num_cols, 3))
            for i in range(num_rows):
                for j in range(num_cols):
                    output[
                    i * self.size:(i + 1) * self.size, j * self.size:(j + 1) * self.size, :
                    ] = np.transpose(img[i * num_cols + j], (1, 2, 0))

            img = output

        img = np.array(Image.fromarray(img.astype(np.uint8)).resize(
            (self.render_args['size'] * num_cols,
             self.render_args['size'] * num_rows)
        ))

        if mode == 'human':
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return img
        else:
            raise ValueError('Render mode not recognised.')

    def _observe_agent(self, agent: int) -> torch.Tensor:
        agents = torch.arange(self.num_snakes)
        layers = {
            (self._food > EPS).squeeze(1): self.food_colour,
            (self._bodies[:, agent].unsqueeze(1) > EPS).squeeze(1): self.self_colour/2,
            (self._heads[:, agent].unsqueeze(1) > EPS).squeeze(1): self.self_colour,
            (self._bodies[:, agents[agents != agent]].sum(dim=1) > EPS): self.other_colour/2,
            (self._heads[:, agents[agents != agent]].sum(dim=1) > EPS): self.other_colour
        }
        return self._make_generic_rgb(layers).to(dtype=self.dtype) / 255

    def _observe(self, mode: str = None) -> Dict[str, torch.Tensor]:
        if mode is None:
            mode = self.observation_mode

        if mode == 'full':
            return {f'agent_{i}': self._observe_agent(i) for i in range(self.num_snakes)}
        elif mode.startswith('partial_'):
            # Get full batch of images
            img = self._get_env_images()

            # Normalise to 0-1
            img = img.float() / 255

            # Crop bits for each agent
            # Pad envs so we ge the correct size observation even when the head of the snake
            # is close to the edge of the environment
            padding = [self.observation_width, self.observation_width, ] * 2
            padded_size = self.size + 2*self.observation_width
            padded_img = F.pad(img, padding)

            dict_observations = {}
            filter = torch.ones((1, 1, self.observation_size, self.observation_size)).to(self.device)
            for i in range(self.num_snakes):
                n_living = (~self.dones[f'agent_{i}']).sum().item()
                head_channel = self.head_channels[i]
                heads = self.envs[:, head_channel:head_channel+1, :, :]
                padded_heads = F.pad(heads, padding)
                head_area_indices = F.conv2d(
                    padded_heads,
                    filter,
                    padding=self.observation_width
                ).round()

                living_observations = padded_img[
                    head_area_indices.expand_as(padded_img).byte()
                ]
                living_observations = living_observations.reshape(
                    n_living, 3, self.observation_size, self.observation_size)

                observations = torch.zeros((self.num_envs, 3, self.observation_size, self.observation_size),
                                           dtype=self.dtype, device=self.device)

                observations[~self.dones[f'agent_{i}']] = living_observations

                dict_observations[f'agent_{i}'] = observations.clone()

            return dict_observations
        else:
            raise ValueError('Unrecognised observation mode.')

    def sanitize_movements(self, movements: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
        mask = orientations == movements
        movements = (movements + (mask * 2).long()).fmod(4)
        return movements

    def _move_heads(self, heads: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """Takes in heads and directions, returns updated heads"""
        # Create head position deltas
        filters = ORIENTATION_FILTERS.to(dtype=self.dtype, device=self.device)
        head_deltas = F.conv2d(
            heads,
            filters,
            padding=1
        )
        directions_onehot = F.one_hot(directions, 4).to(self.dtype)
        head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, directions_onehot]).unsqueeze(1)
        heads += head_deltas
        return heads

    def _update_orientations(self, movements: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
        orientations = (movements + 2).fmod(4)
        return orientations

    def _get_food_overlap(self, heads: torch.Tensor, foods: torch.Tensor) -> torch.Tensor:
        return heads * foods

    def _decay_bodies(self, bodies: torch.Tensor) -> torch.Tensor:
        return (bodies-1).relu()

    def _check_collisions(self, heads: torch.Tensor, pathing: torch.Tensor) -> torch.Tensor:
        return (heads * pathing).gt(EPS)

    def _add_food(self):
        if self.food_mode == 'only_one':
            # Add new food only if there is none in the environment
            food_addition_env_indices = self.foods.view(self.num_envs, -1).sum(dim=-1) < EPS
            if food_addition_env_indices.sum().item() > 0:
                add_food_envs = self.envs[food_addition_env_indices, :, :, :]
                food_addition = self._get_food_addition(add_food_envs)
                self.foods[food_addition_env_indices] += food_addition
        elif self.food_mode == 'random_rate':
            # Have a maximum amount of available food
            food_addition_env_indices = (food(self.envs).view(self.num_envs, -1).sum(dim=-1) < self.max_food)
            n = food_addition_env_indices.sum().item()

            # Get empty locations
            filled_locations = self.envs[food_addition_env_indices].sum(dim=1, keepdim=True) > EPS

            food_addition = torch.rand((n, 1, self.size, self.size), device=self.device)
            # Remove boundaries
            food_addition[:, :, :1, :] = 1
            food_addition[:, :, :, :1] = 1
            food_addition[:, :, -1:, :] = 1
            food_addition[:, :, :, -1:] = 1
            # Each pixel will independently spawn food at a certain rate
            food_addition = food_addition.lt(self.food_rate)

            food_addition &= ~filled_locations

            self.envs[food_addition_env_indices,FOOD_CHANNEL:FOOD_CHANNEL + 1] += food_addition.float()
        else:
            raise ValueError('food_mechanics not recognised')

    def _check_edges(self, heads: torch.Tensor) -> torch.Tensor:
        pass

    def __handle_deaths(self, i: int, agent: str):
        if self.dones[agent].sum() > 0:
            dead_snakes = self.envs[self.dones[agent], self.body_channels[i], :, :].clone()
            # Clear edge locations so we don't spawn food in the edge
            dead_snakes[:, :1, :] = 0
            dead_snakes[:, :, :1] = 0
            dead_snakes[:, -1:, :] = 0
            dead_snakes[:, :, -1:] = 0
            dead_snakes = (dead_snakes.round() > 0).float()

            # Create food at dead snake positions with probability self.food_on_death_prob
            # Don't create food where there's already a snake
            other_snakes = torch.ones(self.num_snakes, dtype=torch.uint8, device=self.device)
            other_snakes[i] = 0
            other_bodies = self._bodies[self.dones[agent]][:, other_snakes, :, :].sum(dim=1) > EPS
            prob = (dead_snakes * torch.rand_like(dead_snakes) > (1 - self.food_on_death_prob))
            food_addition_mask = (
                    prob & ~other_bodies
            ).unsqueeze(1)

            self.envs[self.dones[agent], FOOD_CHANNEL:FOOD_CHANNEL+1] += food_addition_mask.to(self.dtype)
            self.envs[:, 0] = self.envs[:, 0].clamp(0, 1)

            # Remove any snakes that are dead
            self.envs[self.dones[agent], self.body_channels[i], :, :] = 0
            self.envs[self.dones[agent], self.head_channels[i], :, :] = 0

    def _food_from_death(self, dead: torch.Tensor, living: torch.Tensor) -> torch.Tensor:
        """Inputs are dead bodies, outputs"""

    def _boost_costs(self, i: int, agent: str, active_envs: torch.Tensor):
        boost_cost_env_mask = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)
        boost_cost_env_mask[active_envs] = 1
        head_channel = self.head_channels[i]
        body_channel = self.body_channels[i]
        _env = self.envs[active_envs][:, [0, head_channel, body_channel], :, :]
        tail_locations = body(_env) == 1
        self.envs[boost_cost_env_mask, FOOD_CHANNEL] += tail_locations.squeeze(1).to(self.dtype)

        # Decay bodies
        self.envs[boost_cost_env_mask, body_channel:body_channel + 1, :, :] -= 1
        self.envs[boost_cost_env_mask, body_channel:body_channel + 1, :, :] = \
            self.envs[boost_cost_env_mask, body_channel:body_channel + 1, :, :].relu()

        # Reward penalty
        self.rewards[agent][active_envs] -= 1

    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[dict, dict, dict, dict]:
        if len(actions) != self.num_snakes:
            raise RuntimeError('Must have a Tensor of actions for each snake')

        for agent, act in actions.items():
            if act.dtype not in (torch.short, torch.int, torch.long):
                raise TypeError('actions Tensor must be an integer type i.e. '
                                '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

            if act.shape[0] != self.num_envs:
                raise RuntimeError('Must have the same number of actions as environments.')

        # Clear info
        self.info = {}
        move_directions = dict()
        boost_actions = dict()
        for i, (agent, act) in enumerate(actions.items()):
            move_directions[agent] = act.fmod(4)
            boost_actions[agent] = act > 3
            self.info[f'boost_{i}'] = boost_actions[agent].clone()

        # self.rewards = OrderedDict([
        #     (
        #         f'agent_{i}',
        #         torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device).requires_grad_(False)
        #     ) for i in range(self.num_snakes)
        # ])

        snake_sizes = self.bodies.view(self.num_envs*self.num_snakes, -1).max(dim=-1)[0]

        # self.boost_this_step = dict()
        # for i, (agent, act) in enumerate(actions.items()):
        #     self.boost_this_step[agent] = (boost_actions[agent] & (snake_sizes[agent] >= 4))
        #
        # at_least_one_boost = torch.stack([v for k, v in boost_actions.items()]).sum() >= 1
        # at_least_one_size_4 = torch.any(torch.stack([v for k, v in snake_sizes.items()]).sum() >= 4)

        all_envs = torch.ones(self.num_envs, dtype=torch.uint8, device=self.device)
        # if self.boost and at_least_one_boost and at_least_one_size_4:
        #     if self.verbose > 0:
        #         print('>>> Boost phase')
        #     ##############
        #     # Boost step #
        #     ##############
        #     any_boosted = torch.stack([v for k, v in self.boost_this_step.items()]).t().any(dim=1)
        #
        #     # Check orientations and move head positions of all snakes
        #     t0 = time()
        #     for i, (agent, act) in enumerate(actions.items()):
        #         if self.boost_this_step[agent].sum() >= 1:
        #             self._move_heads(i, agent, directions, self.boost_this_step[agent])
        #
        #     self._log(f'Movement: {1000*(time() - t0)}ms')
        #
        #     # Decay bodies of all snakes that haven't eaten food
        #     t0 = time()
        #     food_consumption = {
        #         f'agent_{i}':
        #         torch.zeros((self.num_envs,), dtype=torch.uint8, device=self.device).requires_grad_(False)
        #         for i in range(self.num_snakes)
        #     }
        #     for i, (agent, _) in enumerate(actions.items()):
        #         if self.boost_this_step[agent].sum() >= 1:
        #             self._decay_bodies(i, agent, self.boost_this_step[agent], food_consumption)
        #
        #     self._log(f'Growth/decay: {1000*(time() - t0)}ms')
        #
        #     # Check for collisions with snakes and food
        #     t0 = time()
        #     for i, (agent, _) in enumerate(actions.items()):
        #         if self.boost_this_step[agent].sum() >= 1:
        #             self._check_collisions(i, agent, self.boost_this_step[agent], snake_sizes, food_consumption)
        #     self._log(f'Collision: {1000*(time() - t0)}ms')
        #
        #     # Check for edge collisions
        #     t0 = time()
        #     self._check_all_boundaries(any_boosted)
        #     self._log(f'Edge check: {1000*(time() - t0)}ms')
        #
        #     # Clear dead snakes and create food at dead snakes
        #     t0 = time()
        #     for i, (agent, _) in enumerate(actions.items()):
        #         if self.boost_this_step[agent].sum() >= 1:
        #             self._handle_deaths(i, agent)
        #
        #     self._log(f'Deaths: {1000 * (time() - t0)}ms')
        #
        #     # Handle cost of boost
        #     t0 = time()
        #     for i, (agent, _) in enumerate(actions.items()):
        #         apply_cost = torch.rand(self.num_envs, device=self.device) < self.boost_cost_prob
        #         if (self.boost_this_step[agent] & apply_cost).sum() >= 1:
        #             self._boost_costs(i, agent, self.boost_this_step[agent] & apply_cost)
        #
        #     self._log(f'Boost cost: {1000 * (time() - t0)}ms')
        #
        # snake_sizes = dict()
        # for i, (agent, act) in enumerate(actions.items()):
        #     body_channel = self.body_channels[i]
        #     snake_sizes[agent] = self.envs[:, body_channel:body_channel + 1, :].view(self.num_envs, -1).max(dim=1)[0]
        #
        # # Apply rounding to stop numerical errors accumulating
        # self.envs.round_()

        ################
        # Regular step #
        ################

        self._log('>>> Regular phase')
        # Check orientations and move head positions of all snakes
        t0 = time()
        move_directions = torch.stack([v for k, v in move_directions.items()]).flatten()
        move_directions = self.sanitize_movements(movements=move_directions, orientations=self.orientations)
        self.heads = self._move_heads(self.heads, move_directions)
        self.orientations = self._update_orientations(movements=move_directions, orientations=self.orientations)
        self._log(f'Movement: {1000 * (time() - t0)}ms')

        t0 = time()
        # Get food overlap
        food_overlap = self._get_food_overlap(self.heads, self.foods.repeat_interleave(self.num_snakes, 0))

        # Decay bodies of all snakes that haven't eaten food
        decay_mask = food_overlap.view(self.num_envs*self.num_snakes, -1).sum(dim=-1).lt(EPS)
        self.bodies[decay_mask] = self._decay_bodies(self.bodies[decay_mask])
        self.rewards += (~decay_mask).float()
        self._log(f'Growth/decay: {1000*(time() - t0)}ms')

        # Check for collisions
        other = torch.arange(self.num_snakes, device=self.device).repeat(self.num_envs)
        other = ~F.one_hot(other, self.num_snakes).byte()
        heads = self.heads.view(self.num_envs, self.num_snakes, self.size, self.size).repeat_interleave(self.num_snakes, 0)
        bodies = self.bodies.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True).expand_as(self.bodies)
        pathing = torch.einsum('nshw,ns->nhw', [heads, other.float()]).unsqueeze(1)
        pathing += bodies
        collisions = self._check_collisions(self.heads, pathing)
        self.dones |= collisions.view(self.num_envs*self.num_snakes, -1).any(dim=-1)

        # Move bodies forward
        body_growth = food_overlap.view(self.num_envs * self.num_snakes, -1).sum(dim=-1)
        self.bodies += self.heads * (snake_sizes + body_growth)[:, None, None, None]

        # Create food at dead snake locations



        # # Check for collisions with snakes and food
        # t0 = time()
        # self._check_all_collisions(all_envs, snake_sizes, food_consumption)
        # self._log(f'Collision: {1000*(time() - t0)}ms')
        #
        # # Check for edge collisions
        # t0 = time()
        # self._check_all_boundaries(all_envs)
        # self._log(f'Edge check: {1000*(time() - t0)}ms')
        #
        # # Clear dead snakes and create food at dead snakes
        # t0 = time()
        # # self._handle_all_deaths()
        # for i, (agent, _) in enumerate(actions.items()):
        #     self._handle_deaths(i, agent)

        # self._log(f'Deaths: {1000 * (time() - t0)}ms')

        # # Add food if there is none in the environment
        # self._add_food()

        # # Apply rounding to stop numerical errors accumulating
        # self.foods.round_()
        # self.heads.round_()
        # self.bodies.round_()

        # # Environment is finished if all snake are dead
        # self.dones['__all__'] = torch.ones(self.num_envs, dtype=torch.uint8, device=self.device)
        # for agent, _ in actions.items():
        #     self.dones['__all__'] &= self.dones[agent]
        #
        # # or if its past the maximum episode length
        # self.dones['__all__'] |= self.env_lifetimes > self.max_env_lifetime

        # Get observations
        t0 = time()
        observations = {}
        self._log(f'Observations: {1000 * (time() - t0)}ms')

        dones = {f'agent_{i}': d for i, d in enumerate(self.dones.clone().view(self.num_snakes, self.num_envs).unbind())}
        rewards = {f'agent_{i}': d for i, d in enumerate(self.rewards.clone().view(self.num_snakes, self.num_envs).unbind())}

        return observations, rewards, dones, self.info

    def check_consistency(self):
        """Runs multiple checks for environment consistency and throws an exception if any fail"""
        n = self.num_envs
        return

        for i in range(self.num_snakes):
            head_channel = self.head_channels[i]
            body_channel = self.body_channels[i]

            # Check dead snakes all 0
            if self.dones[f'agent_{i}'].sum() > 0:
                dead_envs = torch.cat([
                    self.envs[self.dones[f'agent_{i}'], head_channel:head_channel+1:, :, :],
                    self.envs[self.dones[f'agent_{i}'], body_channel:body_channel + 1, :, :]
                ], dim=1)
                dead_all_zeros = dead_envs.sum() == 0
                if not dead_all_zeros:
                    raise RuntimeError(f'Dead snake (agent_{i}) contains non-zero elements.')

            # Check living envs
            if (~self.dones[f'agent_{i}']).sum() > 0:
                living_envs = torch.cat([
                    self.envs[~self.dones[f'agent_{i}'], 0:1, :, :],
                    self.envs[~self.dones[f'agent_{i}'], head_channel:head_channel + 1, :, :],
                    self.envs[~self.dones[f'agent_{i}'], body_channel:body_channel + 1, :, :]
                ], dim=1)
                try:
                    snake_consistency(living_envs)
                except RuntimeError as e:
                    print(f'agent_{i}')
                    raise e

            if self.food_mode == 'only_one':
                # Environment contains one food instance
                contains_one_food = torch.all(food(self.envs).view(n, -1).sum(dim=-1) >= 1)
                if not contains_one_food:
                    raise RuntimeError('An environment doesn\'t contain at least one food instance')

        # Check no overlapping snakes
        # Sum of bodies in each square of each env should be no more than 1
        body_exists = (self._bodies < EPS).float()
        no_overlapping_bodies = torch.all(body_exists.view(n, -1).max(dim=1)[0] <= 1)
        if not no_overlapping_bodies:
            raise RuntimeError('An environment contains overlapping snakes')

        # Check number of heads is no more than env.num_snakes
        num_heads = self._heads.sum(dim=1, keepdim=True).view(n, -1)
        if not torch.all(num_heads <= self.num_snakes):
            raise RuntimeError('An environment contains more snakes than it should.')

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
            food_indices.t(),
            torch.ones(len(food_indices)),
            available_locations.shape,
            device=self.device,
            dtype=self.dtype
        )
        food_addition = food_addition.to_dense()

        return food_addition

    def reset(self, done: torch.Tensor = None):
        """Resets environments in which the snake has died

        Args:
            done: A 1D Tensor of length self.num_envs. A value of 1 means the corresponding environment needs to be
                reset
        """
        # Reset envs that contain no snakes
        t0 = time()
        if done is None:
            done = self.dones['__all__']

        done = done.view((done.shape[0]))
        num_done = int(done.sum().item())

        # Create environments
        if done.sum() > 0:
            new_envs, new_positions = self._create_envs(num_done)
            self.envs[done.byte(), :, :, :] = new_envs
            self.orientations[done] = new_positions

        # Reset done trackers
        self.dones['__all__'][done] = 0
        self.env_lifetimes[done] = 0
        for i in range(self.num_snakes):
            self.dones[f'agent_{i}'][done] = 0

        if self.verbose:
            print(f'Resetting {num_done} envs: {1000 * (time() - t0)}ms')

        # Optionally respawn snakes
        if self.respawn_mode == 'any':
            t0 = time()
            num_respawned = 0
            for i in range(self.num_snakes):
                if torch.any(self.dones[f'agent_{i}']):
                    # Add snake
                    successfully_respawned = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)
                    respawned_envs, respawns, new_positions = self._add_snake(
                        envs=self.envs[self.dones[f'agent_{i}']],
                        snake_channel=i,
                        exception_on_failure=False
                    )
                    self.envs[self.dones[f'agent_{i}']] = respawned_envs
                    num_respawned += respawns.sum().item()
                    successfully_respawned[self.dones[f'agent_{i}']] = respawns
                    # Reset done trackers
                    self.dones[f'agent_{i}'][successfully_respawned] = 0

            if self.verbose:
                print(f'Respawned {num_respawned} snakes: {1000 * (time() - t0)}ms')

        return self._observe()

    def _add_snake(self,
                   envs: torch.Tensor,
                   snake_channel: int,
                   exception_on_failure: bool) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Adds a snake in a certain channel to environments.

        Args:
            envs: Tensor represening the environments.
            snake_channel: Which snake to add
            exception_on_failure: If True then raise an exception if there is no available spawning locations
        """
        l = self.initial_snake_length - 1
        n = envs.shape[0]

        successfully_spawned = torch.zeros(n, dtype=torch.uint8, device=self.device)

        occupied_locations = envs.sum(dim=1, keepdim=True) > EPS
        # Expand this because you can't put a snake right next to another snake
        occupied_locations = (F.conv2d(
            occupied_locations.to(dtype=self.dtype),
            torch.ones((1, 1, 3, 3), dtype=self.dtype, device=self.device),
            padding=1
        ) > EPS).byte()

        available_locations = (envs.sum(dim=1, keepdim=True) < EPS) & ~occupied_locations

        # Remove boundaries
        available_locations[:, :, :l, :] = 0
        available_locations[:, :, :, :l] = 0
        available_locations[:, :, -l:, :] = 0
        available_locations[:, :, :, -l:] = 0

        any_available_locations = available_locations.view(n, -1).max(dim=1)[0].byte()
        if exception_on_failure:
            # If there is no available locations for a snake raise an exception
            if torch.any(~any_available_locations):
                raise RuntimeError('There is no available locations to create snake!')

        successfully_spawned |= any_available_locations

        body_seed_indices = drop_duplicates(torch.nonzero(available_locations), 0)
        # Body seeds is a tensor that contains all zeros except where a snake will be spawned
        # Shape: (n, 1, self.size, self.size)
        body_seeds = torch.sparse_coo_tensor(
            body_seed_indices.t(), torch.ones(len(body_seed_indices)), available_locations.shape,
            device=self.device, dtype=self.dtype
        )

        # Choose random starting directions
        random_directions = torch.randint(4, (n,), device=self.device)
        random_directions_onehot = torch.empty((n, 4), dtype=self.dtype, device=self.device)
        random_directions_onehot.zero_()
        random_directions_onehot.scatter_(1, random_directions.unsqueeze(-1), 1)

        # Create bodies
        bodies = torch.einsum('bchw,bc->bhw', [
            F.conv2d(
                body_seeds.to_dense(),
                LENGTH_3_SNAKES.to(self.device).to(dtype=self.dtype),
                padding=1
            ),
            random_directions_onehot
        ])
        envs[:, self.body_channels[snake_channel], :, :] = bodies

        # Create heads at end of bodies
        snake_sizes = envs[:, self.body_channels[snake_channel], :].view(n, -1).max(dim=1)[0]
        # Only create heads where there is a snake. This catches an edge case where there is no room
        # for a snake to spawn and hence snake size == bodies everywhere (as bodies is all 0)
        snake_sizes[snake_sizes == 0] = -1
        snake_size_mask = snake_sizes[:, None, None].expand((n, self.size, self.size))
        envs[:, self.head_channels[snake_channel], :, :] = (bodies == snake_size_mask).to(dtype=self.dtype)

        # Start tracking head positions and orientations
        new_positions = torch.zeros((n, 3), dtype=torch.long, device=self.device)
        new_positions[:, 0] = random_directions
        heads = envs[:, self.head_channels[snake_channel], :, :]
        locations = torch.nonzero(heads)[:, 1:]
        dones = ~heads.view(n, -1).sum(dim=1).gt(EPS)
        new_positions[~dones, 1:] = \
            torch.nonzero(envs[:, self.head_channels[snake_channel], :, :])[:, 1:]

        return envs, successfully_spawned, new_positions

    def _create_envs(self, num_envs: int) -> (torch.Tensor, torch.Tensor):
        """Vectorised environment creation. Creates self.num_envs environments simultaneously."""
        if self.initial_snake_length != 3:
            raise NotImplementedError('Only initial snake length = 3 has been implemented.')

        envs = torch.zeros((num_envs, 1 + 2 * self.num_snakes, self.size, self.size), dtype=self.dtype, device=self.device)

        new_positions = []
        for i in range(self.num_snakes):
            envs, _, _new_positions = self._add_snake(envs, i, True)
            new_positions.append(_new_positions)

        new_positions = torch.stack(new_positions, dim=1)

        # Add food
        food_addition = self._get_food_addition(envs)
        envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_addition

        return envs.round(), new_positions
