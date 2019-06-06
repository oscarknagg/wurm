from time import time, sleep
from typing import Dict, Tuple, Optional
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
                 reward_on_death: int = -1,
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
        self.boost_this_step = torch.zeros(self.num_envs * self.num_snakes, dtype=torch.uint8, device=self.device)
        self.rewards = torch.zeros(self.num_envs * self.num_snakes, dtype=torch.float, device=self.device)
        self.env_lifetimes = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.snake_lifetimes = torch.zeros((num_envs, num_snakes), dtype=torch.long, device=device)
        self.orientations = torch.zeros((num_envs*num_snakes), dtype=torch.long, device=self.device, requires_grad=False)
        self.viewer = 0

        if not manual_setup:
            # Create environments
            (self.foods, self.heads, self.bodies), self.orientations = self._create_envs(self.num_envs)
            self.foods.requires_grad_(False)
            self.heads.requires_grad_(False)
            self.bodies.requires_grad_(False)

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
        # self.max_food = self.num_snakes * 10
        self.max_env_lifetime = 5000
        self.reward_on_death = reward_on_death

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

        self.agent_colours = self.get_n_colours(self.num_envs*self.num_snakes)
        self.num_colours = self.agent_colours.shape[0]

        self.info = {}

        self.edge_locations_mask = torch.zeros(
            (1, 1, self.size, self.size), dtype=self.dtype, device=self.device)

        self.edge_locations_mask[:, :, :1, :] = 1
        self.edge_locations_mask[:, :, :, :1] = 1
        self.edge_locations_mask[:, :, -1:, :] = 1
        self.edge_locations_mask[:, :, :, -1:] = 1

    def get_n_colours(self, n: int) -> torch.Tensor:
        colours = torch.rand((n, 3), device=self.device)
        colours[:, 0] /= 1.5  # Reduce red
        colours /= colours.norm(2, dim=1, keepdim=True)
        colours *= 192
        colours = colours.short()
        return colours

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
        agent_colours = self.agent_colours

        intensity_factor = (self.bodies.gt(EPS).float() * 1 / 3) + (self.heads.gt(EPS).float() * 1 / 3)
        intensity_factor *= (1 + 0.5*self.boost_this_step.float())[:, None, None, None].expand_as(intensity_factor)
        intensity_factor = intensity_factor.squeeze(1)

        body_colours = torch.einsum('nhw,nc->nchw', [intensity_factor, agent_colours.float()])

        img = body_colours\
            .reshape(self.num_envs, self.num_snakes, 3, self.size, self.size)\
            .sum(dim=1)\
            .short()

        food_colours = torch.einsum('nihw,ic->nchw', [self.foods.gt(EPS).float(), self.food_colour.unsqueeze(0).float()])
        img += food_colours.short()

        # Convert to NHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))

        black_colour = torch.zeros(3, device=self.device, dtype=torch.short)
        black_colour = black_colour[None, None, None, :]
        white_colour = torch.ones(3, device=self.device, dtype=torch.short) * 255

        black_colour_mask = (img == black_colour).all(dim=-1, keepdim=True)
        img[black_colour_mask.squeeze(-1), :] = white_colour

        # Convert back to NCHW axes
        img = img.permute((0, 3, 1, 2))

        # Black boundaries
        img[self.edge_locations_mask.expand_as(img).byte()] = 0

        return img

    def render(self, mode: str = 'human', env: int = None):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        img = self._get_env_images()
        # Convert to numpy
        img = img.cpu().numpy()

        # Rearrange images depending on number of envs
        if self.num_envs == 1 or env is not None:
            num_cols = num_rows = 1
            img = img[env or 0]
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

        bodies_per_env = self.bodies.view(self.num_envs, self.num_snakes, self.size, self.size)
        heads_per_env = self.heads.view(self.num_envs, self.num_snakes, self.size, self.size)

        layers = {
            self.foods.gt(EPS).squeeze(1): self.food_colour,
            bodies_per_env[:, agent].gt(EPS): self.self_colour/2,
            heads_per_env[:, agent].gt(EPS): self.self_colour,
            bodies_per_env[:, agents[agents != agent]].sum(dim=1).gt(EPS): self.other_colour / 2,
            heads_per_env[:, agents[agents != agent]].sum(dim=1).gt(EPS): self.other_colour,
        }
        return self._make_generic_rgb(layers).to(dtype=self.dtype) / 255

    def _observe(self, mode: str = None) -> Dict[str, torch.Tensor]:
        if mode is None:
            mode = self.observation_mode

        if mode == 'full':
            return OrderedDict([(f'agent_{i}', self._observe_agent(i)) for i in range(self.num_snakes)])
        elif mode.startswith('partial_'):
            # Get full batch of images
            t0 = time()
            img = self._get_env_images()
            self._log(f'Rendering: {1000 * (time() - t0)}ms')

            # Normalise to 0-1
            img = img.float() / 255

            # Crop bits for each agent
            # Pad envs so we ge the correct size observation even when the head of the snake
            # is close to the edge of the environment
            padding = [self.observation_width, self.observation_width, ] * 2
            padded_img = F.pad(img, padding).repeat_interleave(self.num_snakes, dim=0)

            t0 = time()
            n_living = (~self.dones).sum().item()
            filter = torch.ones((1, 1, self.observation_size, self.observation_size)).to(self.device)
            padded_heads = F.pad(self.heads, padding)
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

            observations = torch.zeros((self.num_envs*self.num_snakes, 3, self.observation_size, self.observation_size),
                                       dtype=self.dtype, device=self.device)

            observations[~self.dones] = living_observations
            observations = observations\
                .reshape(self.num_envs, self.num_snakes, 3, self.observation_size, self.observation_size)

            self._log(f'Head cropping: {1000 * (time() - t0)}ms')

            dict_observations = OrderedDict([
                (f'agent_{i}', observations[:, i].clone()) for i in range(self.num_snakes)
            ])
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

    def _update_orientations(self, movements: torch.Tensor) -> torch.Tensor:
        orientations = (movements + 2).fmod(4)
        return orientations

    def _get_food_overlap(self, heads: torch.Tensor, foods: torch.Tensor) -> torch.Tensor:
        return (heads * foods).gt(EPS)

    def _decay_bodies(self, bodies: torch.Tensor) -> torch.Tensor:
        return (bodies-1).relu()

    def _check_collisions(self, heads: torch.Tensor, pathing: torch.Tensor) -> torch.Tensor:
        return (heads * pathing).gt(EPS)

    def _add_food(self):
        if self.food_mode == 'only_one':
            # Add new food only if there is none in the environment
            food_addition_env_indices = self.foods.view(self.num_envs, -1).sum(dim=-1) < EPS
            n = food_addition_env_indices.sum().item()
            if n > 0:
                food_addition = self._get_food_addition(
                    heads=self.heads[food_addition_env_indices.repeat_interleave(self.num_snakes)],
                    bodies=self.bodies[food_addition_env_indices.repeat_interleave(self.num_snakes)],
                    foods=self.foods[food_addition_env_indices],
                )
                self.foods[food_addition_env_indices] += food_addition
        elif self.food_mode == 'random_rate':
            # Have a maximum amount of available food
            food_addition_env_indices = (self.foods.view(self.num_envs, -1).sum(dim=-1) < self.max_food)
            n = food_addition_env_indices.sum().item()

            # Get empty locations
            _envs = torch.cat([
                self.foods[food_addition_env_indices],
                self.heads[food_addition_env_indices.repeat_interleave(self.num_snakes)].view(n, self.num_snakes, self.size, self.size),
                self.bodies[food_addition_env_indices.repeat_interleave(self.num_snakes)].view(n, self.num_snakes, self.size, self.size)
            ], dim=1)

            # Get empty locations
            available_locations = _envs.sum(dim=1, keepdim=True) < EPS
            # Remove boundaries
            available_locations[:, :, :1, :] = 0
            available_locations[:, :, :, :1] = 0
            available_locations[:, :, -1:, :] = 0
            available_locations[:, :, :, -1:] = 0
            filled_locations = ~available_locations

            food_addition = torch.rand((n, 1, self.size, self.size), device=self.device)
            # Each pixel will independently spawn food at a certain rate
            food_addition = food_addition.lt(self.food_rate)

            food_addition &= ~filled_locations
            if torch.any(food_addition == 2) or torch.any(food_addition < 0):
                raise RuntimeError('Bad food_addition')
            self.foods[food_addition_env_indices] += food_addition.float()
        else:
            raise ValueError('food_mechanics not recognised')

    def _check_edges(self, heads: torch.Tensor) -> torch.Tensor:
        edge_collision = (heads * self.edge_locations_mask.expand_as(heads))
        return edge_collision

    def _food_from_death(self, dead: torch.Tensor, living: torch.Tensor) -> torch.Tensor:
        """Inputs are dead bodies, outputs"""
        dead[:, :, 1, :] = 0
        dead[:, :, :, :1] = 0
        dead[:, :, -1:, :] = 0
        dead[:, :, :, -1:] = 0
        dead = (dead.round() > 0).float()

        prob = (dead * torch.rand_like(dead) > (1 - self.food_on_death_prob))

        food_addition_mask = prob & ~living

        return food_addition_mask

    def _get_food_addition(self, heads: torch.Tensor, bodies: torch.Tensor, foods: torch.Tensor):
        n = foods.size(0)
        _envs = torch.cat([
            foods,
            # foods.repeat_interleave(self.num_snakes, dim=0),
            heads.view(n, self.num_snakes, self.size, self.size),
            bodies.view(n, self.num_snakes, self.size, self.size)
        ], dim=1)

        # Get empty locations
        available_locations = _envs.sum(dim=1, keepdim=True) < EPS
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

    def _to_per_agent_dict(self, tensor: torch.Tensor, key: str):
        return {f'{key}_{i}': d for i, d in enumerate(tensor.clone().view(self.num_envs, self.num_snakes).t().unbind())}

    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], dict, dict, dict]:
        if len(actions) != self.num_snakes:
            raise RuntimeError('Must have a Tensor of actions for each snake')

        for agent, act in actions.items():
            if act.dtype not in (torch.short, torch.int, torch.long):
                raise TypeError('actions Tensor must be an integer type i.e. '
                                '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

            if act.shape[0] != self.num_envs:
                raise RuntimeError('Must have the same number of actions as environments.')

        # Clear info
        snake_collision_tracker = torch.zeros(self.num_envs * self.num_snakes, dtype=torch.uint8, device=self.device)
        edge_collision_tracker = torch.zeros(self.num_envs * self.num_snakes, dtype=torch.uint8, device=self.device)
        food_consumption_tracker = torch.zeros(self.num_envs * self.num_snakes, dtype=self.dtype, device=self.device)
        self.rewards = torch.zeros(self.num_envs * self.num_snakes, dtype=torch.float, device=self.device)
        self.info = {}
        move_directions = dict()
        boost_actions = dict()
        for i, (agent, act) in enumerate(actions.items()):
            move_directions[agent] = act.fmod(4)
            boost_actions[agent] = act > 3
            self.info[f'boost_{i}'] = boost_actions[agent].clone()
            self.info[f'snake_collision_{i}'] = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)
            self.info[f'edge_collision_{i}'] = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)

        snake_sizes = self.bodies.view(self.num_envs*self.num_snakes, -1).max(dim=-1)[0]
        done_at_start = self.dones.clone()

        move_directions = torch.stack([v for k, v in move_directions.items()]).t().flatten()
        move_directions = self.sanitize_movements(movements=move_directions, orientations=self.orientations)
        self.orientations = self._update_orientations(movements=move_directions)

        boost_actions = torch.stack([v for k, v in boost_actions.items()]).t().flatten()
        size_gte_4 = snake_sizes >= 4
        boosted_agents = boost_actions & size_gte_4
        self.boost_this_step = boosted_agents
        n_boosted = boosted_agents.sum().item()
        boosted_envs = boosted_agents.view(self.num_envs, self.num_snakes).any(dim=1)
        n_boosted_envs = boosted_envs.sum().item()
        if self.boost and torch.any(boosted_agents):
            self._log('>>>Boost phase')
            ##############
            # Boost step #
            ##############
            t0 = time()
            self.heads[boosted_agents] = self._move_heads(self.heads[boosted_agents], move_directions[boosted_agents])
            self._log(f'Movement: {1000 * (time() - t0)}ms')

            t0 = time()
            # Get food overlap
            food_overlap = self._get_food_overlap(self.heads, self.foods.repeat_interleave(self.num_snakes, 0)).float()
            # Clamp here because if two snakes simultaneiously move their heads on to a food pixel then we get a food
            # overlap pixel with a value of 2 (which then leads to a food pixel of -1 and some issue down the line)
            self.foods -= food_overlap \
                .view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True).clamp(0, 1)
            self._log(f'Food overlap: {1000 * (time() - t0)}ms')

            t0 = time()
            # Decay bodies of all snakes that haven't eaten food
            decay_mask = boosted_agents.clone()
            decay_mask[boosted_agents] &= food_overlap[boosted_agents].view(n_boosted, -1).sum(dim=-1).lt(EPS)
            self.bodies[decay_mask] = \
                self._decay_bodies(self.bodies[decay_mask])
            eaten_food = food_overlap[boosted_agents].view(n_boosted, -1).sum(dim=-1).gt(EPS).float()
            self.rewards[boosted_agents] += eaten_food
            food_consumption_tracker[boosted_agents] += eaten_food
            self._log(f'Growth/decay: {1000 * (time() - t0)}ms')

            t0 = time()
            # Check for collisions
            other = torch.arange(self.num_snakes, device=self.device).repeat(self.num_envs)
            other = ~F.one_hot(other, self.num_snakes).byte()
            heads = self.heads \
                .view(self.num_envs, self.num_snakes, self.size, self.size) \
                .repeat_interleave(self.num_snakes, 0)
            bodies = self.bodies \
                .view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True) \
                .repeat_interleave(self.num_snakes, dim=0)
            pathing = torch.einsum('nshw,ns->nhw', [heads, other.float()]).unsqueeze(1)
            pathing += bodies
            collisions = self._check_collisions(self.heads[boosted_agents], pathing[boosted_agents])\
                .view(n_boosted, -1).any(dim=-1)
            self.dones[boosted_agents] |= collisions
            snake_collision_tracker[boosted_agents] |= collisions
            self._log(f'Collisions: {1000 * (time() - t0)}ms')

            t0 = time()
            # Move bodies forward
            body_growth = food_overlap[boosted_agents].view(n_boosted, -1).sum(dim=-1).gt(EPS).float()
            self.bodies[boosted_agents] += self.heads[boosted_agents] * (snake_sizes[boosted_agents] + body_growth)[:, None, None, None]
            # Update snake sizes
            snake_sizes[boosted_agents] += body_growth
            self._log(f'Move bodies: {1000 * (time() - t0)}ms')

            t0 = time()
            # Check for edge collisions
            edge_collisions = self._check_edges(self.heads[boosted_agents]).view(n_boosted, -1).gt(EPS).any(dim=-1)
            self.dones[boosted_agents] |= edge_collisions
            edge_collision_tracker[boosted_agents] |= edge_collisions
            self._log(f'Edges: {1000 * (time() - t0)}ms')

            if self.food_on_death_prob > 0:
                t0 = time()
                # Create food at dead snake locations
                _bodies = self.bodies.clone()
                _bodies[~self.dones] = 0
                dead = _bodies.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True)
                _bodies = self.bodies.clone()
                _bodies[self.dones] = 0
                living = _bodies.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True).gt(EPS)
                food_on_death = self._food_from_death(dead, living).float()
                self.foods += food_on_death
                self._log(f'Deaths: {1000 * (time() - t0)}ms')

            # Apply boost cost
            boost_cost_agents = boosted_agents & (torch.rand(self.num_envs*self.num_snakes, device=self.device) < self.boost_cost_prob)
            if boost_cost_agents.sum() > 0:
                t0 = time()
                boost_cost_envs = boost_cost_agents.view(self.num_envs, self.num_snakes).any(dim=1)
                tail_locations = self.bodies == 1
                tail_locations[~boost_cost_agents] = 0
                self.foods += tail_locations\
                    .view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True).gt(EPS).float()

                self.bodies[boost_cost_agents] = \
                    self._decay_bodies(self.bodies[boost_cost_agents])
                self.rewards -= boost_cost_agents.float()
                snake_sizes[boost_cost_agents] -= 1
                self._log(f'Boost cost: {1000 * (time() - t0)}ms')

            # Delete done snakes
            self.bodies[self.dones] = 0
            self.heads[self.dones] = 0

            # Apply rounding to stop numerical errors accumulating
            self.foods.round_()
            # Occasionally we end up with a food pixel of value 2, probably because two food addition events
            # have occured in one place twice in the same .step(). As I can't figure out why this happens I've
            # added this quick fix
            self.foods.clamp_(0, 1)
            self.heads.round_()
            self.bodies.round_()

        ################
        # Regular step #
        ################
        self._log('>>> Regular phase')
        # Check orientations and move head positions of all snakes
        t0 = time()
        self.heads = self._move_heads(self.heads, move_directions)
        self._log(f'Movement: {1000 * (time() - t0)}ms')

        t0 = time()
        # Get food overlap
        food_overlap = self._get_food_overlap(self.heads, self.foods.repeat_interleave(self.num_snakes, 0)).float()
        # Clamp here because if two snakes simultaneiously move their heads on to a food pixel then we get a food
        # overlap pixel with a value of 2 (which then leads to a food pixel of -1 and some issue down the line)
        food_overlap = food_overlap
        self.foods -= food_overlap.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True).clamp(0, 1)
        self._log(f'Food overlap: {1000 * (time() - t0)}ms')

        t0 = time()
        # Decay bodies of all snakes that haven't eaten food
        decay_mask = food_overlap.view(self.num_envs*self.num_snakes, -1).sum(dim=-1).lt(EPS)
        self.bodies[decay_mask] = self._decay_bodies(self.bodies[decay_mask])
        eaten_food = food_overlap.view(self.num_envs*self.num_snakes, -1).sum(dim=-1).gt(EPS).float()
        self.rewards += eaten_food
        food_consumption_tracker += eaten_food
        self._log(f'Growth/decay: {1000*(time() - t0)}ms')

        t0 = time()
        # Check for collisions
        other = torch.arange(self.num_snakes, device=self.device).repeat(self.num_envs)
        other = ~F.one_hot(other, self.num_snakes).byte()
        heads = self.heads.view(self.num_envs, self.num_snakes, self.size, self.size).repeat_interleave(self.num_snakes, 0)
        bodies = self.bodies.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True).repeat_interleave(self.num_snakes, dim=0)
        pathing = torch.einsum('nshw,ns->nhw', [heads, other.float()]).unsqueeze(1)
        pathing += bodies
        collisions = self._check_collisions(self.heads, pathing).view(self.num_envs*self.num_snakes, -1).any(dim=-1)
        self.dones |= collisions
        snake_collision_tracker |= collisions
        self._log(f'Collisions: {1000 * (time() - t0)}ms')

        t0 = time()
        # Move bodies forward
        body_growth = food_overlap.view(self.num_envs * self.num_snakes, -1).sum(dim=-1).gt(EPS).float()
        self.bodies += self.heads * (snake_sizes + body_growth)[:, None, None, None]
        # Update snake sizes
        snake_sizes += body_growth
        self._log(f'Move bodies: {1000 * (time() - t0)}ms')

        t0 = time()
        # Check for edge collisions
        edge_collisions = self._check_edges(self.heads).view(self.num_envs * self.num_snakes, -1).gt(EPS).any(dim=-1)
        self.dones |= edge_collisions
        edge_collision_tracker |= edge_collisions
        self._log(f'Edges: {1000 * (time() - t0)}ms')

        if self.food_on_death_prob > 0:
            t0 = time()
            # Create food at dead snake locations
            _bodies = self.bodies.clone()
            _bodies[~self.dones] = 0
            dead = _bodies.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True)
            _bodies = self.bodies.clone()
            _bodies[self.dones] = 0
            living = _bodies.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True).gt(EPS)
            food_on_death = self._food_from_death(dead, living).float()
            self.foods += food_on_death
            self._log(f'Deaths: {1000 * (time() - t0)}ms')

        # Delete done snakes
        self.bodies[self.dones] = 0
        self.heads[self.dones] = 0

        # Add food if there is none in the environment
        self._add_food()

        # Give negative reward on death
        died_this_step = self.dones & (~done_at_start)
        death_rewards = died_this_step.float() * self.reward_on_death
        self.rewards += death_rewards

        # Apply rounding to stop numerical errors accumulating
        self.foods.round_()
        # Occasionally we end up with a food pixel of value 2, probably because two food addition events
        # have occured in one place twice in the same .step(). As I can't figure out why this happens I've
        # added this quick fix
        self.foods.clamp_(0, 1)
        self.heads.round_()
        self.bodies.round_()

        # Get observations
        t0 = time()
        observations = self._observe()
        self._log(f'Observations: {1000 * (time() - t0)}ms')

        dones = {f'agent_{i}': d for i, d in enumerate(self.dones.clone().view(self.num_envs, self.num_snakes).t().unbind())}
        # # Environment is finished if all snake are dead
        dones['__all__'] = self.dones.view(self.num_envs, self.num_snakes).all(dim=1).clone()
        # # or if its past the maximum episode length
        dones['__all__'] |= self.env_lifetimes > self.max_env_lifetime

        rewards = {f'agent_{i}': d for i, d in enumerate(self.rewards.clone().view(self.num_envs, self.num_snakes).t().unbind())}

        # Update info
        self.info.update({
            f'snake_collision_{i}': d for i, d in
            enumerate(snake_collision_tracker.clone().view(self.num_envs, self.num_snakes).t().unbind())
        })
        self.info.update({
            f'edge_collision_{i}': d for i, d in
            enumerate(edge_collision_tracker.clone().view(self.num_envs, self.num_snakes).t().unbind())
        })
        self.info.update({
            f'food_{i}': d for i, d in
            enumerate(food_consumption_tracker.clone().view(self.num_envs, self.num_snakes).t().unbind())
        })
        self.info.update({
            f'boost_{i}': d for i, d in
            enumerate(self.boost_this_step.clone().view(self.num_envs, self.num_snakes).t().unbind())
        })
        self.info.update({
            f'size_{i}': d for i, d in
            enumerate(snake_sizes.clone().view(self.num_envs, self.num_snakes).t().unbind())
        })

        return observations, rewards, dones, self.info

    def check_consistency(self):
        """Runs multiple checks for environment consistency and throws an exception if any fail"""
        _envs = torch.cat([
            self.foods.repeat_interleave(self.num_snakes, dim=0),
            self.heads,
            self.bodies
        ], dim=1)

        living_envs = _envs[~self.dones]
        snake_consistency(living_envs.round())

        # Check no overlapping snakes
        # Sum of bodies in each square of each env should be no more than 1
        overlapping_bodies = self.bodies.view(self.num_envs, self.num_snakes, self.size, self.size)\
            .gt(EPS)\
            .sum(dim=1, keepdim=True)\
            .view(self.num_envs, -1)\
            .max(dim=-1)[0]\
            .gt(1)
        if torch.any(overlapping_bodies):
            for i in self.bodies[overlapping_bodies.repeat_interleave(self.num_snakes, 0)][:self.num_snakes]:
                print(i)
            print('-'*10)
            _overlapping = self.bodies.view(self.num_envs, self.num_snakes, self.size, self.size).gt(EPS).sum(dim=1, keepdim=True)
            print(_overlapping[overlapping_bodies][0])
            raise RuntimeError('An environment contains overlapping snakes')

        # Check number of heads is no more than env.num_snakes
        num_heads = self.heads.view(self.num_envs, self.num_snakes, self.size, self.size) \
            .sum(dim=1, keepdim=True)
        if not torch.all(num_heads <= self.num_snakes):
            raise RuntimeError('An environment contains more snakes than it should.')

        dead_envs = _envs[self.dones]
        dead_all_zeros = dead_envs[:, 1:].sum() == 0
        if not dead_all_zeros:
            raise RuntimeError(f'Dead snake contains non-zero elements.')

    def reset(self, done: torch.Tensor = None, return_observations: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        """Resets environments in which the snake has died

        Args:
            done: A 1D Tensor of length self.num_envs. A value of 1 means the corresponding environment needs to be
                reset
        """
        # Reset envs that contain no snakes
        if done is None:
            done = self.dones.view(self.num_envs, self.num_snakes).all(dim=1)

        agent_dones = done.repeat_interleave(self.num_snakes)
        done = done.view((done.shape[0]))
        num_done = int(done.sum().item())

        # Create environments
        if done.sum() > 0:
            t0 = time()
            (new_foods, new_heads, new_bodies), new_positions = self._create_envs(num_done)
            self.foods[done] = new_foods
            self.heads[agent_dones] = new_heads
            self.bodies[agent_dones] = new_bodies
            self.orientations[agent_dones] = new_positions
            self._log(f'Resetting {num_done} envs: {1000 * (time() - t0)}ms')

        # Reset done trackers
        self.env_lifetimes[done] = 0
        self.dones[agent_dones] = 0

        # Change agent colours each death
        new_colours = self.get_n_colours(self.dones.sum().item())
        self.agent_colours[self.dones] = new_colours

        if self.respawn_mode == 'any':
            if torch.any(self.dones):
                t0 = time()
                any_done_in_env = self.dones.view(self.num_envs, self.num_snakes).any(dim=1)

                # Only spawn the first dead snake from each env
                # i.e. only spawn one snake per step
                first_done_per_env = (self.dones.view(self.num_envs, self.num_snakes).cumsum(dim=1) == 1).flatten() & self.dones
                _envs = torch.cat([
                    self.foods.repeat_interleave(self.num_snakes, 0),
                    self.heads,
                    self.bodies,
                ], dim=1).sum(dim=1, keepdim=True)
                _envs = _envs.view(self.num_envs, self.num_snakes, self.size, self.size).sum(dim=1, keepdim=True).repeat_interleave(self.num_snakes, 0)
                _envs = _envs[first_done_per_env]

                # Get empty locations
                pathing = _envs.sum(dim=1, keepdim=True) > EPS
                new_bodies, new_heads, new_orientations, successfully_spawned = self._get_snake_addition(
                    pathing, exception_on_failure=False)

                self.bodies[first_done_per_env] = new_bodies
                self.heads[first_done_per_env] = new_heads
                self.orientations[first_done_per_env] = new_orientations
                self.dones[first_done_per_env] = ~successfully_spawned

                self._log(f'Respawned {successfully_spawned.sum().item()} snakes: {1000 * (time() - t0)}ms')

        if return_observations:
            observations = self._observe()

            return observations

    def _get_snake_addition(self,
                            pathing: torch.Tensor,
                            exception_on_failure: bool) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """pathing is a nx1xhxw tensor containing filled locations"""
        n = pathing.shape[0]
        l = self.initial_snake_length - 1
        successfully_spawned = torch.zeros(n, dtype=torch.uint8, device=self.device)

        # Expand pathing because you can't put a snake right next to another snake
        t0 = time()
        pathing = (F.conv2d(
            pathing.to(dtype=self.dtype),
            torch.ones((1, 1, 3, 3), dtype=self.dtype, device=self.device),
            padding=1
        ) > EPS).byte()
        # Remove boundaries
        pathing[:, :, :l, :] = 1
        pathing[:, :, :, :l] = 1
        pathing[:, :, -l:, :] = 1
        pathing[:, :, :, -l:] = 1
        available_locations = ~pathing
        self._log(f'Respawn location calculation: {1000 * (time() - t0)}ms')

        any_available_locations = available_locations.view(n, -1).max(dim=1)[0].byte()
        if exception_on_failure:
            # If there is no available locations for a snake raise an exception
            if torch.any(~any_available_locations):
                raise RuntimeError('There is no available locations to create snake!')

        successfully_spawned |= any_available_locations

        t0 = time()
        body_seed_indices = drop_duplicates(torch.nonzero(available_locations), 0)
        # Body seeds is a tensor that contains all zeros except where a snake will be spawned
        # Shape: (n, 1, self.size, self.size)
        body_seeds = torch.sparse_coo_tensor(
            body_seed_indices.t(), torch.ones(len(body_seed_indices)), available_locations.shape,
            device=self.device, dtype=self.dtype
        )
        self._log(f'Choose spawn locations: {1000 * (time() - t0)}ms')

        t0 = time()
        # Choose random starting directions
        random_directions = torch.randint(4, (n,), device=self.device)
        random_directions_onehot = torch.empty((n, 4), dtype=self.dtype, device=self.device)
        random_directions_onehot.zero_()
        random_directions_onehot.scatter_(1, random_directions.unsqueeze(-1), 1)
        self._log(f'Choose spawn directions: {1000 * (time() - t0)}ms')

        t0 = time()
        # Create bodies
        new_bodies = torch.einsum('bchw,bc->bhw', [
            F.conv2d(
                body_seeds.to_dense(),
                LENGTH_3_SNAKES.to(self.device).to(dtype=self.dtype),
                padding=1
            ),
            random_directions_onehot
        ]).unsqueeze(1)
        self._log(f'Create bodies: {1000 * (time() - t0)}ms')

        t0 = time()
        # Create heads at end of bodies
        snake_sizes = new_bodies.view(n, -1).max(dim=1)[0]
        # Only create heads where there is a snake. This catches an edge case where there is no room
        # for a snake to spawn and hence snake size == bodies everywhere (as bodies is all 0)
        snake_sizes[snake_sizes == 0] = -1
        snake_size_mask = snake_sizes[:, None, None, None].expand((n, 1, self.size, self.size))
        new_heads = (new_bodies == snake_size_mask).to(dtype=self.dtype)
        self._log(f'Create heads: {1000 * (time() - t0)}ms')

        return new_bodies, new_heads, random_directions, successfully_spawned

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
        envs[:, 2*(snake_channel+1), :, :] = bodies
        # envs[:, self.body_channels[snake_channel], :, :] = bodies

        # Create heads at end of bodies
        snake_sizes = envs[:, 2*(snake_channel+1), :].view(n, -1).max(dim=1)[0]
        # Only create heads where there is a snake. This catches an edge case where there is no room
        # for a snake to spawn and hence snake size == bodies everywhere (as bodies is all 0)
        snake_sizes[snake_sizes == 0] = -1
        snake_size_mask = snake_sizes[:, None, None].expand((n, self.size, self.size))
        envs[:, 2*(snake_channel+1) - 1, :, :] = (bodies == snake_size_mask).to(dtype=self.dtype)

        # Start tracking head positions and orientations
        new_positions = torch.zeros((n, 3), dtype=torch.long, device=self.device)
        new_positions[:, 0] = random_directions
        heads = envs[:, 2*(snake_channel+1) - 1, :, :]
        locations = torch.nonzero(heads)[:, 1:]
        dones = ~heads.view(n, -1).sum(dim=1).gt(EPS)
        new_positions[~dones, 1:] = \
            torch.nonzero(envs[:, 2*(snake_channel+1) - 1, :, :])[:, 1:]

        return envs, successfully_spawned, new_positions

    def _create_envs(self, num_envs: int) -> (Tuple[torch.Tensor], torch.Tensor):
        """Vectorised environment creation. Creates self.num_envs environments simultaneously."""
        if self.initial_snake_length != 3:
            raise NotImplementedError('Only initial snake length = 3 has been implemented.')

        envs = torch.zeros((num_envs, 1 + 2 * self.num_snakes, self.size, self.size), dtype=self.dtype, device=self.device)

        new_positions = []
        for i in range(self.num_snakes):
            envs, _, _new_positions = self._add_snake(envs, i, True)
            new_positions.append(_new_positions)

        new_positions = torch.stack(new_positions, dim=1)

        # Slice and reshape to get foods, heads and bodies
        foods = envs[:, 0:1].round()
        heads = envs[:, [2*(i+1)-1 for i in range(self.num_snakes)]].view(num_envs*self.num_snakes, 1, self.size, self.size).round()
        bodies = envs[:, [2*(i+1) for i in range(self.num_snakes)]].view(num_envs*self.num_snakes, 1, self.size, self.size).round()

        # Add food
        food_addition = self._get_food_addition(foods=foods, heads=heads, bodies=bodies)
        foods += food_addition

        return (foods, heads, bodies), new_positions[:, :, 0].reshape(num_envs*self.num_snakes)
