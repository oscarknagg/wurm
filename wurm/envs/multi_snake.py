from time import time
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
                 observation_mode: str = 'one_channel',
                 device: str = DEFAULT_DEVICE,
                 dtype: torch.dtype = torch.float,
                 manual_setup: bool = False,
                 food_on_death_prob: float = 0.5,
                 boost: bool = True,
                 boost_cost_prob: float = 0.5,
                 food_mechanics: str = 'only_one',
                 food_rate: float = None,
                 respawn_mode: str = 'all',
                 verbose: int = 0,
                 render_args: dict = None):
        """Initialise the environments

        Args:
            num_envs:
            size:
            max_timesteps:
        """
        self.num_envs = num_envs
        self.num_snakes = num_snakes
        self.size = size
        self.initial_snake_length = initial_snake_length
        self.on_death = on_death
        self.observation_mode = observation_mode
        self.device = device
        self.verbose = verbose
        self.dtype = dtype

        if render_args is None:
            self.render_args = {'num_rows': 1, 'num_cols': 1, 'size': 256}
        else:
            self.render_args = render_args

        self.envs = torch.zeros((num_envs, 1 + 2 * num_snakes, size, size), dtype=self.dtype, device=self.device).requires_grad_(False)
        self.head_channels = [1 + 2*i for i in range(self.num_snakes)]
        self.body_channels = [2 + 2*i for i in range(self.num_snakes)]
        self.dones = {f'agent_{i}': torch.zeros(num_envs, dtype=torch.uint8, device=self.device) for i in range(self.num_snakes)}
        self.dones.update({'__all__': torch.zeros(num_envs, dtype=torch.uint8, device=self.device)})
        self.t = 0
        self.viewer = 0

        if not manual_setup:
            # Create environments
            self.envs = self._create_envs(self.num_envs)
            self.envs.requires_grad_(False)

        ###################################
        # Environment dynamics parameters #
        ###################################
        self.respawn_mode = respawn_mode
        self.food_on_death_prob = food_on_death_prob
        self.boost = boost
        self.boost_cost_prob = boost_cost_prob
        self.food_mechanics = food_mechanics
        self.food_rate = 3/(2 * (size - 2)) if food_rate is None else food_rate

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

        agent_colours = torch.rand(size=(num_snakes, 3), device=self.device)
        agent_colours /= agent_colours.norm(2, dim=1, keepdim=True)
        agent_colours *= 192
        self.agent_colours = agent_colours.short()

        self.info = {}
        self.rewards = {}

        self.boost_this_step = OrderedDict([
            (
                f'agent_{i}',
                torch.zeros((self.num_envs,), dtype=torch.uint8, device=self.device).requires_grad_(False)
            ) for i in range(self.num_snakes)
        ])

        self.edge_locations_mask = torch.zeros(
            (1, 1, self.size, self.size), dtype=self.dtype, device=self.device)

        self.edge_locations_mask[:, :, :1, :] = 1
        self.edge_locations_mask[:, :, :, :1] = 1
        self.edge_locations_mask[:, :, -1:, :] = 1
        self.edge_locations_mask[:, :, :, -1:] = 1

    def _log(self, msg: str):
        if self.verbose > 0:
            print(msg)

    def _make_generic_rgb(self, colour_layers: Dict[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        img = torch.ones((self.num_envs, 3, self.size, self.size), dtype=torch.short, device=self.device) * 255

        # Convert to BHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))

        for locations, colour in colour_layers.items():
            img[locations, :] = colour

        # print()
        img[:, :1, :, :] = self.edge_colour
        img[:, :, :1, :] = self.edge_colour
        img[:, -1:, :, :] = self.edge_colour
        img[:, :, -1:, :] = self.edge_colour

        # Convert back to BCHW axes
        img = img.permute((0, 3, 1, 2))

        return img

    def render(self, mode: str = 'human'):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        # Regular snakes
        layers = {
            self._food.gt(EPS).squeeze(1): self.food_colour,
        }

        for i, (agent, has_boosted) in enumerate(self.boost_this_step.items()):
            if has_boosted.sum() > 0:
                boosted_bodies = torch.zeros((self.num_envs, self.size, self.size), dtype=torch.uint8, device=self.device)
                boosted_heads = torch.zeros((self.num_envs, self.size, self.size), dtype=torch.uint8, device=self.device)
                layers.update({
                    boosted_bodies: (self.agent_colours[i].float() * (2 / 3)).short(),
                    boosted_heads: (self.agent_colours[i].float() * (4 / 3)).short(),
                })

            regular_bodies = torch.zeros((self.num_envs, self.size, self.size), dtype=torch.uint8, device=self.device)
            regular_heads = torch.zeros((self.num_envs, self.size, self.size), dtype=torch.uint8, device=self.device)
            regular_bodies[~has_boosted] = self._bodies[~has_boosted, i:i+1].sum(dim=1).gt(EPS)
            regular_heads[~has_boosted] = self._heads[~has_boosted, i:i+1].sum(dim=1).gt(EPS)
            layers.update({
                regular_bodies: (self.agent_colours[i].float() * (2 / 3)).short(),
                regular_heads: (self.agent_colours[i].float() * (4 / 3)).short(),
            })

        # Get RBG Tensor NCHW
        img = self._make_generic_rgb(layers)

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

    def _observe(self):
        return {f'agent_{i}': self._observe_agent(i) for i in range(self.num_snakes)}

    @property
    def _food(self) -> torch.Tensor:
        return self.envs[:, 0:1, :, :]

    @property
    def _heads(self) -> torch.Tensor:
        return self.envs[:, self.head_channels, :, :]

    @property
    def _bodies(self) -> torch.Tensor:
        return self.envs[:, self.body_channels, :, :]

    def _move_heads(self, i: int, agent: str, directions: Dict[str, torch.Tensor], active_envs: torch.Tensor):
        # The sub-environment of just one agent
        head_channel = self.head_channels[i]
        body_channel = self.body_channels[i]
        _env = self.envs[active_envs][:, [0, head_channel, body_channel], :, :]
        t0 = time()
        orientations = determine_orientations(_env)
        self._log(f'-Orientations: {1000*(time()-t0)}ms')

        # Check if this snake is trying to move backwards and change
        # it's direction/action to just continue forward
        # The test for this is if their orientation number {0, 1, 2, 3}
        # is the same as their action
        t0 = time()
        mask = orientations == directions[agent][active_envs]
        directions[agent][active_envs] = (directions[agent][active_envs] + (mask * 2).long()).fmod(4)
        self._log(f'-Sanitize directions: {1000*(time()-t0)}ms')

        t0 = time()
        # Create head position deltas
        head_deltas = F.conv2d(
            head(_env),
            ORIENTATION_FILTERS.to(dtype=self.dtype, device=self.device),
            padding=1
        )
        # Select the head position delta corresponding to the correct action
        directions_onehot = F.one_hot(directions[agent][active_envs], 4).to(self.dtype)
        head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, directions_onehot]).unsqueeze(1)

        # Move head position by applying delta
        self.envs[active_envs, head_channel:head_channel + 1, :, :] += head_deltas.round()
        self._log(f'-Head convs: {1000*(time()-t0)}ms')

    def _decay_bodies(self, i: int, agent: str, active_envs: torch.Tensor, food_consumption: dict):
        n = active_envs.sum().item()
        # Decay only active envs
        body_decay_env_mask = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)
        body_decay_env_mask[active_envs] = 1

        head_channel = self.head_channels[i]
        body_channel = self.body_channels[i]
        _env = self.envs[active_envs][:, [0, head_channel, body_channel], :, :]

        head_food_overlap = (head(_env) * food(_env)).view(n, -1).sum(dim=-1)

        # Decay the body sizes by 1, hence moving the body, apply ReLu to keep above 0
        # Only do this for environments which haven't just eaten food
        body_decay_env_mask[active_envs] = ~head_food_overlap.byte()

        food_consumption[agent] = ~body_decay_env_mask

        self.envs[body_decay_env_mask, body_channel:body_channel + 1, :, :] -= 1
        self.envs[body_decay_env_mask, body_channel:body_channel + 1, :, :] = \
            self.envs[body_decay_env_mask, body_channel:body_channel + 1, :, :].relu()

    def _check_collisions(self, i: int, agent: str, active_envs: torch.Tensor, snake_sizes: dict, food_consumption: dict):
        n = active_envs.sum().item()
        # Check if any snakes have collided with themselves or any other snakes
        head_channel = self.head_channels[i]
        body_channel = self.body_channels[i]
        _env = self.envs[active_envs][:, [0, head_channel, body_channel], :, :]

        # Collision with body of any snake
        bodies = self._bodies[active_envs].sum(dim=1, keepdim=True)
        body_collision = (head(_env) * bodies).view(n, -1).sum(dim=-1) > EPS

        other_snakes = torch.ones(self.num_snakes, dtype=torch.uint8, device=self.device)
        other_snakes[i] = 0
        other_heads = self._heads[:, other_snakes, :, :]
        head_collision = (head(_env) * other_heads[active_envs]).view(n, -1).sum(dim=-1) > EPS
        snake_collision = body_collision | head_collision

        if f'snake_collision_{i}' not in self.info.keys():
            self.info[f'snake_collision_{i}'] = torch.zeros((self.num_envs,), dtype=torch.uint8,
                                                            device=self.device).requires_grad_(False)
        self.info[f'snake_collision_{i}'][active_envs] |= snake_collision
        self.dones[agent][active_envs] |= snake_collision

        # Create a new head position in the body channel
        # Make this head +1 greater if the snake has just eaten food
        self.envs[active_envs, body_channel:body_channel + 1, :, :] += \
            head(_env) * (
                    snake_sizes[agent][active_envs, None, None, None].expand((n, 1, self.size, self.size)) +
                    food_consumption[agent][active_envs, None, None, None].expand((n, 1, self.size, self.size)).to(self.dtype)
            )

        food_removal = head(_env) * food(_env) * -1
        self.rewards[agent][active_envs] -= (food_removal.view(n, -1).sum(dim=-1).to(self.dtype))
        self.envs[active_envs, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_removal

    def _add_food(self):
        if self.food_mechanics == 'only_one':
            # Add new food only if there is none in the environment
            food_addition_env_indices = (food(self.envs).view(self.num_envs, -1).sum(dim=-1) < EPS)
        elif self.food_mechanics == 'random_rate':
            # Add new food with a certain probability
            food_addition_env_indices = torch.rand((self.num_envs), device=self.device) < self.food_rate
        else:
            raise ValueError('food_mechanics not recognised')

        if food_addition_env_indices.sum().item() > 0:
            add_food_envs = self.envs[food_addition_env_indices, :, :, :]
            food_addition = self._get_food_addition(add_food_envs)
            self.envs[food_addition_env_indices, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_addition

    def _check_boundaries(self, i: int, agent: str, active_envs: torch.Tensor):
        n = active_envs.sum()
        head_channel = self.head_channels[i]
        body_channel = self.body_channels[i]
        _env = self.envs[active_envs][:, [0, head_channel, body_channel], :, :]
        edge_collision = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)

        heads = head(_env)
        edge_collision[active_envs] |= (heads * self.edge_locations_mask.expand_as(heads)).view(n, -1).sum(dim=-1) > EPS

        self.dones[agent] |= edge_collision
        head_exists = self._heads[:, i].view(self.num_envs, -1).max(dim=-1)[0] > EPS
        edge_collision = edge_collision & head_exists
        if f'edge_collision_{i}' not in self.info.keys():
            self.info[f'edge_collision_{i}'] = torch.zeros((self.num_envs,), dtype=torch.uint8,
                                                            device=self.device).requires_grad_(False)
        self.info[f'edge_collision_{i}'] |= edge_collision

    def _handle_deaths(self, i: int, agent: str):
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
        directions = dict()
        boost_actions = dict()
        for i, (agent, act) in enumerate(actions.items()):
            directions[agent] = act.fmod(4)
            boost_actions[agent] = act > 3
            self.info[f'boost_{i}'] = boost_actions[agent].clone()

        self.rewards = OrderedDict([
            (
                f'agent_{i}',
                torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device).requires_grad_(False)
            ) for i in range(self.num_snakes)
        ])

        snake_sizes = dict()
        for i, (agent, act) in enumerate(actions.items()):
            body_channel = self.body_channels[i]
            snake_sizes[agent] = self.envs[:, body_channel:body_channel + 1, :].view(self.num_envs, -1).max(dim=1)[0]

        self.boost_this_step = dict()
        for i, (agent, act) in enumerate(actions.items()):
            self.boost_this_step[agent] = (boost_actions[agent] & (snake_sizes[agent] >= 4))

        at_least_one_boost = torch.stack([v for k, v in boost_actions.items()]).sum() >= 1
        at_least_one_size_4 = torch.any(torch.stack([v for k, v in snake_sizes.items()]).sum() >= 4)

        all_envs = torch.ones(self.num_envs, dtype=torch.uint8, device=self.device)
        if self.boost and at_least_one_boost and at_least_one_size_4:
            if self.verbose > 0:
                print('>>> Boost phase')
            ##############
            # Boost step #
            ##############
            # Check orientations and move head positions of all snakes
            t0 = time()
            for i, (agent, act) in enumerate(actions.items()):
                if self.boost_this_step[agent].sum() >= 1:
                    self._move_heads(i, agent, directions, self.boost_this_step[agent])

            self._log(f'Movement: {1000*(time() - t0)}ms')

            # Decay bodies of all snakes that haven't eaten food
            t0 = time()
            food_consumption = dict()
            for i, (agent, _) in enumerate(actions.items()):
                if self.boost_this_step[agent].sum() >= 1:
                    self._decay_bodies(i, agent, self.boost_this_step[agent], food_consumption)

            self._log(f'Growth/decay: {1000*(time() - t0)}ms')

            # Check for collisions with snakes and food
            t0 = time()
            for i, (agent, _) in enumerate(actions.items()):
                if self.boost_this_step[agent].sum() >= 1:
                    self._check_collisions(i, agent, self.boost_this_step[agent], snake_sizes, food_consumption)
                    # self._check_collisions(i, agent, all_envs, snake_sizes, food_consumption)

            self._log(f'Collision: {1000*(time() - t0)}ms')

            # Check for edge collisions
            t0 = time()
            for i, (agent, _) in enumerate(actions.items()):
                if self.boost_this_step[agent].sum() >= 1:
                    self._check_boundaries(i, agent, self.boost_this_step[agent])

            self._log(f'Edge check: {1000*(time() - t0)}ms')

            # Clear dead snakes and create food at dead snakes
            t0 = time()
            for i, (agent, _) in enumerate(actions.items()):
                if self.boost_this_step[agent].sum() >= 1:
                    self._handle_deaths(i, agent)

            self._log(f'Deaths: {1000 * (time() - t0)}ms')

            # Handle cost of boost
            t0 = time()
            for i, (agent, _) in enumerate(actions.items()):
                apply_cost = torch.rand(self.num_envs, device=self.device) < self.boost_cost_prob
                if (self.boost_this_step[agent] & apply_cost).sum() >= 1:
                    self._boost_costs(i, agent, self.boost_this_step[agent] & apply_cost)

            self._log(f'Boost cost: {1000 * (time() - t0)}ms')

            # Add food if there is none in the environment
            self._add_food()

        snake_sizes = dict()
        for i, (agent, act) in enumerate(actions.items()):
            body_channel = self.body_channels[i]
            snake_sizes[agent] = self.envs[:, body_channel:body_channel + 1, :].view(self.num_envs, -1).max(dim=1)[0]

        # Apply rounding to stop numerical errors accumulating
        self.envs.round_()

        ################
        # Regular step #
        ################
        if self.verbose > 0:
            print('>>> Regular phase')
        # Check orientations and move head positions of all snakes
        t0 = time()
        for i, (agent, act) in enumerate(actions.items()):
            self._move_heads(i, agent, directions, all_envs)

        self._log(f'Movement: {1000 * (time() - t0)}ms')

        # Decay bodies of all snakes that haven't eaten food
        t0 = time()
        food_consumption = dict()
        for i, (agent, _) in enumerate(actions.items()):
            self._decay_bodies(i, agent, all_envs, food_consumption)

        self._log(f'Growth/decay: {1000*(time() - t0)}ms')

        # Check for collisions with snakes and food
        t0 = time()
        for i, (agent, _) in enumerate(actions.items()):
            self._check_collisions(i, agent, all_envs, snake_sizes, food_consumption)

        self._log(f'Collision: {1000*(time() - t0)}ms')

        # Check for edge collisions
        t0 = time()
        for i, (agent, _) in enumerate(actions.items()):
            self._check_boundaries(i, agent, all_envs)

        self._log(f'Edge check: {1000*(time() - t0)}ms')

        # Clear dead snakes and create food at dead snakes
        t0 = time()
        for i, (agent, _) in enumerate(actions.items()):
            self._handle_deaths(i, agent)

        self._log(f'Deaths: {1000 * (time() - t0)}ms')

        # Add food if there is none in the environment
        self._add_food()

        # Apply rounding to stop numerical errors accumulating
        self.envs.round_()

        # Environment is finished if all snake are dead
        self.dones['__all__'] = torch.ones(self.num_envs, dtype=torch.uint8, device=self.device)
        for agent, _ in actions.items():
            self.dones['__all__'] &= self.dones[agent]

        return self._observe(), self.rewards, {k: v.clone() for k, v in self.dones.items()}, self.info

    def check_consistency(self):
        """Runs multiple checks for environment consistency and throws an exception if any fail"""
        n = self.num_envs

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
            new_envs = self._create_envs(num_done)
            self.envs[done.byte(), :, :, :] = new_envs

        # Reset done trackers
        self.dones['__all__'][done] = 0
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
                    respawned_envs, respawns = self._add_snake(
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
                   exception_on_failure: bool) -> (torch.Tensor, torch.Tensor):
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

        return envs, successfully_spawned

    def _create_envs(self, num_envs: int) -> torch.Tensor:
        """Vectorised environment creation. Creates self.num_envs environments simultaneously."""
        if self.initial_snake_length != 3:
            raise NotImplementedError('Only initial snake length = 3 has been implemented.')

        envs = torch.zeros((num_envs, 1 + 2 * self.num_snakes, self.size, self.size), dtype=self.dtype, device=self.device)

        for i in range(self.num_snakes):
            envs, _ = self._add_snake(envs, i, True)

            # Add food
        food_addition = self._get_food_addition(envs)
        envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_addition

        # Add food
        food_addition = self._get_food_addition(envs)
        envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_addition

        return envs.round()
