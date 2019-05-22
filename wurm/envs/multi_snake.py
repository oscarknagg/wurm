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
                 boost: bool = False,
                 boost_cost_prob: float = 0.5,
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
        self.snake_dones = torch.zeros((num_envs, num_snakes), dtype=torch.uint8, device=self.device)
        self.t = 0
        self.viewer = 0

        if not manual_setup:
            # Create environments
            self.envs = self._create_envs(self.num_envs)
            self.envs.requires_grad_(False)

        self.env_dones = torch.zeros(num_envs, dtype=torch.uint8, device=self.device)
        self.dead = torch.zeros((num_envs, num_snakes), dtype=torch.uint8, device=self.device)

        # Environment dynamics parameters
        self.respawn_mode = ['all_snakes', 'any_snake'][0]
        self.food_on_death_prob = food_on_death_prob
        self.boost = boost
        self.boost_cost_prob = boost_cost_prob

        # Rendering parameters
        self.viewer = None
        # Own snake appears green
        self.self_body_colour = torch.tensor((0, 255 * 0.5, 0), dtype=torch.short, device=self.device)
        self.self_head_colour = torch.tensor((0, 255, 0), dtype=torch.short, device=self.device)
        # Other snakes appear blue
        self.other_body_colour = torch.tensor((0, 0, 255 * 0.5), dtype=torch.short, device=self.device)
        self.other_head_colour = torch.tensor((0, 0, 255), dtype=torch.short, device=self.device)
        self.food_colour = torch.tensor((255, 0, 0), dtype=torch.short, device=self.device)
        self.edge_colour = torch.tensor((0, 0, 0), dtype=torch.short, device=self.device)

        self.info = {}
        self.rewards = {}
        self.dones = {}

    def _make_rgb(self,
                  foods: torch.Tensor,
                  heads: torch.Tensor,
                  bodies: torch.Tensor,
                  other_heads: torch.Tensor = None,
                  other_bodies: torch.Tensor = None):
        img = torch.ones((self.num_envs, 3, self.size, self.size), dtype=torch.short, device=self.device) * 255

        # Convert to BHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))

        objects_to_colours = {
            foods: self.food_colour,
            bodies: self.self_body_colour,
            other_bodies: self.other_body_colour,
            heads: self.self_head_colour,
            other_heads: self.other_head_colour,
        }

        for obj, colour in objects_to_colours.items():
            if obj is not None:
                locations = (obj > EPS).squeeze(1)
                img[locations, :] = colour

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

        # Get RBG Tensor BCHW
        img = self._make_rgb(
            foods=self._food,
            bodies=self._bodies.sum(dim=1 , keepdim=True),
            heads=self._heads.sum(dim=1 , keepdim=True)
        )

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
        return self._make_rgb(
            foods=self._food,
            bodies=self._bodies[:, agent].unsqueeze(1),
            heads=self._heads[:, agent].unsqueeze(1),
            other_bodies=self._bodies[:, agents[agents != agent]],
            other_heads=self._heads[:, agents[agents != agent]]
        ).float() / 255

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
        if self.verbose > 0:
            print(f'-Orientations: {time()-t0}s')

        # Check if this snake is trying to move backwards and change
        # it's direction/action to just continue forward
        # The test for this is if their orientation number {0, 1, 2, 3}
        # is the same as their action
        t0 = time()
        mask = orientations == directions[agent][active_envs]
        directions[agent][active_envs] = (directions[agent][active_envs] + (mask * 2).long()).fmod(4)
        if self.verbose > 0:
            print(f'-Sanitize directions: {time()-t0}s')

        t0 = time()
        # Create head position deltas
        head_deltas = F.conv2d(
            # head(_env).to(dtype=self.dtype),
            head(_env),
            ORIENTATION_FILTERS.to(dtype=self.dtype, device=self.device),
            padding=1
        )
        # Select the head position delta corresponding to the correct action
        directions_onehot = F.one_hot(directions[agent][active_envs], 4).to(self.dtype)
        head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, directions_onehot]).unsqueeze(1)

        # Move head position by applying delta
        self.envs[active_envs, head_channel:head_channel + 1, :, :] += head_deltas.round()
        if self.verbose > 0:
            print(f'-Head convs: {time()-t0}s')

    def _decay_bodies(self, i: int, agent: str, active_envs: torch.Tensor, food_consumption: dict):
        n = active_envs.sum().item()
        body_decay_env_mask = torch.ones(self.num_envs, dtype=torch.uint8, device=self.device)

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
        body_collision = (head(_env) * self._bodies[active_envs]).view(n, -1).sum(dim=-1) > EPS

        # Collision with head of other snake
        other_snakes = torch.ones(self.num_snakes, dtype=torch.uint8, device=self.device)
        other_snakes[i] = 0
        other_heads = self._heads[:, other_snakes, :, :]
        head_collision = (head(_env) * other_heads[active_envs]).view(n, -1).sum(dim=-1) > EPS
        snake_collision = body_collision | head_collision

        self.info.update({f'snake_collision_{i}': snake_collision})
        self.dones[agent][active_envs] |= snake_collision

        # Create a new head position in the body channel
        # Make this head +1 greater if the snake has just eaten food
        self.envs[active_envs, body_channel:body_channel + 1, :, :] += \
            head(_env) * (
                    snake_sizes[agent][active_envs, None, None, None].expand((n, 1, self.size, self.size)) +
                    food_consumption[agent][:, None, None, None].expand((n, 1, self.size, self.size)).to(self.dtype)
            )

        food_removal = head(_env) * food(_env) * -1
        self.rewards[agent][active_envs] -= (food_removal.view(n, -1).sum(dim=-1).to(self.dtype))
        self.envs[active_envs, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_removal

    def _add_food(self):
        # Add new food if necessary.
        food_addition_env_indices = (food(self.envs).view(self.num_envs, -1).sum(dim=-1) < EPS)
        if food_addition_env_indices.sum().item() > 0:
            add_food_envs = self.envs[food_addition_env_indices, :, :, :]
            food_addition = self._get_food_addition(add_food_envs)
            self.envs[food_addition_env_indices, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_addition

    def _check_boundaries(self, i: int, agent: str, active_envs: torch.Tensor):
        # Check for boundary, Done by performing a convolution with no padding
        # If the head is at the edge then it will be cut off and the sum of the head
        # channel will be 0
        n = active_envs.sum()
        head_channel = self.head_channels[i]
        body_channel = self.body_channels[i]
        _env = self.envs[active_envs][:, [0, head_channel, body_channel], :, :]
        edge_collision = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)
        edge_collision[active_envs] |= F.conv2d(
            head(_env),
            NO_CHANGE_FILTER.float().to(dtype=self.dtype, device=self.device),
        ).view(n, -1).sum(dim=-1) < EPS
        self.dones[agent] |= edge_collision
        head_exists = self._heads[:, i].view(self.num_envs, -1).max(dim=-1)[0] > EPS
        edge_collision = edge_collision & head_exists
        self.info.update({f'edge_collision_{i}': edge_collision})

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
            other_bodies = self._bodies[self.dones[agent], other_snakes, :, :] > EPS
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

    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[dict, dict, dict, dict]:
        if len(actions) != self.num_snakes:
            raise RuntimeError('Must have a Tensor of actions for each snake')

        for agent, act in actions.items():
            if act.dtype not in (torch.short, torch.int, torch.long):
                raise TypeError('actions Tensor must be an integer type i.e. '
                                '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

            if act.shape[0] != self.num_envs:
                raise RuntimeError('Must have the same number of actions as environments.')

        directions = dict()
        boosts = dict()
        for i, (agent, act) in enumerate(actions.items()):
            directions[agent] = act.fmod(4)
            boosts[agent] = act > 3

        self.rewards = OrderedDict([
            (
                f'agent_{i}',
                torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device).requires_grad_(False)
            ) for i in range(self.num_snakes)
        ])
        self.dones = OrderedDict([
            (
                f'agent_{i}',
                torch.zeros((self.num_envs,), dtype=torch.uint8, device=self.device).requires_grad_(False)
            ) for i in range(self.num_snakes)
        ])

        # Clear info
        self.info = {}
        snake_sizes = dict()
        for i, (agent, act) in enumerate(actions.items()):
            body_channel = self.body_channels[i]
            snake_sizes[agent] = self.envs[:, body_channel:body_channel + 1, :].view(self.num_envs, -1).max(dim=1)[0]

        at_least_one_boost = torch.stack([v for k, v in boosts.items()]).sum() >= 1
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
                if (boosts[agent] & (snake_sizes[agent] >= 4)).sum() >= 1:
                    self._move_heads(i, agent, directions, boosts[agent] & (snake_sizes[agent] >= 4))

            if self.verbose > 0:
                print(f'Movement: {time() - t0}s')

            # Decay bodies of all snakes that haven't eaten food
            t0 = time()
            food_consumption = dict()
            for i, (agent, _) in enumerate(actions.items()):
                if (boosts[agent] & (snake_sizes[agent] >= 4)).sum() >= 1:
                    self._decay_bodies(i, agent, boosts[agent] & (snake_sizes[agent] >= 4), food_consumption)

            if self.verbose > 0:
                print(f'Growth/decay: {time() - t0}s')

            # Check for collisions with snakes and food
            t0 = time()
            for i, (agent, _) in enumerate(actions.items()):
                if (boosts[agent] & (snake_sizes[agent] >= 4)).sum() >= 1:
                    self._check_collisions(i, agent, all_envs, snake_sizes, food_consumption)

            if self.verbose > 0:
                print(f'Snake collision: {time() - t0}s')

            # Check for edge collisions
            t0 = time()
            for i, (agent, _) in enumerate(actions.items()):
                if (boosts[agent] & (snake_sizes[agent] >= 4)).sum() >= 1:
                    self._check_boundaries(i, agent, boosts[agent] & (snake_sizes[agent] >= 4))

            if self.verbose > 0:
                print(f'Edge collision: {time() - t0}s')

            # Clear dead snakes and create food at dead snakes
            t0 = time()
            for i, (agent, _) in enumerate(actions.items()):
                if (boosts[agent] & (snake_sizes[agent] >= 4)).sum() >= 1:
                    self._handle_deaths(i, agent)

            if self.verbose > 0:
                print(f'Deaths: {time() - t0}s')

            # Handle cost of boost
            t0 = time()
            for i, (agent, _) in enumerate(actions.items()):
                apply_cost = torch.rand(self.num_envs, device=self.device) < self.boost_cost_prob
                if (boosts[agent] & (snake_sizes[agent] >= 4) & apply_cost).sum() >= 1:
                    self._boost_costs(i, agent, boosts[agent] & (snake_sizes[agent] >= 4) & apply_cost)

            if self.verbose > 0:
                print(f'Boost cost: {time() - t0}s')

            # Add food if there is none in the environment
            self._add_food()

        # Environment is finished if all snake are dead
        self.dones['__all__'] = torch.ones((self.num_envs,), dtype=torch.uint8, device=self.device).requires_grad_(False)
        for agent, _ in actions.items():
            self.dones['__all__'] = self.dones['__all__'] & self.dones[agent]

        self.env_dones = self.dones['__all__'].clone()
        for i in range(self.num_snakes):
            self.snake_dones[:, i] = self.snake_dones[:, i] | self.dones[f'agent_{i}']

        snake_sizes = dict()
        for i, (agent, act) in enumerate(actions.items()):
            body_channel = self.body_channels[i]
            snake_sizes[agent] = self.envs[:, body_channel:body_channel + 1, :].view(self.num_envs, -1).max(dim=1)[0]

        # Apply rounding to stop numerical errors accumulating
        self.envs.round_()
        self.check_consistency()

        ################
        # Regular step #
        ################
        if self.verbose > 0:
            print('>>> Regular phase')
        # Check orientations and move head positions of all snakes
        t0 = time()
        for i, (agent, act) in enumerate(actions.items()):
            self._move_heads(i, agent, directions, all_envs)

        if self.verbose > 0:
            print(f'Movement: {time() - t0}s')

        # Decay bodies of all snakes that haven't eaten food
        t0 = time()
        food_consumption = dict()
        for i, (agent, _) in enumerate(actions.items()):
            self._decay_bodies(i, agent, all_envs, food_consumption)

        if self.verbose > 0:
            print(f'Growth/decay: {time() - t0}s')

        # Check for collisions with snakes and food
        t0 = time()
        for i, (agent, _) in enumerate(actions.items()):
            self._check_collisions(i, agent, all_envs, snake_sizes, food_consumption)

        if self.verbose > 0:
            print(f'Snake collision: {time() - t0}s')

        # Check for edge collisions
        t0 = time()
        for i, (agent, _) in enumerate(actions.items()):
            self._check_boundaries(i, agent, all_envs)

        if self.verbose > 0:
            print(f'Edge collision: {time() - t0}s')

        # Clear dead snakes and create food at dead snakes
        t0 = time()
        for i, (agent, _) in enumerate(actions.items()):
            self._handle_deaths(i, agent)

        if self.verbose > 0:
            print(f'Deaths: {time() - t0}s')

        # Add food if there is none in the environment
        self._add_food()

        # Apply rounding to stop numerical errors accumulating
        self.envs.round_()

        # Environment is finished if all snake are dead
        self.dones['__all__'] = torch.ones((self.num_envs,), dtype=torch.uint8, device=self.device).requires_grad_(False)
        for agent, _ in actions.items():
            self.dones['__all__'] = self.dones['__all__'] & self.dones[agent]

        self.env_dones = self.dones['__all__'].clone()
        for i in range(self.num_snakes):
            self.snake_dones[:, i] = self.snake_dones[:, i] | self.dones[f'agent_{i}']

        return self._observe(), self.rewards, self.dones, self.info

    def check_consistency(self):
        """Runs multiple checks for environment consistency and throws an exception if any fail"""
        n = self.num_envs

        for i in range(self.num_snakes):
            head_channel = self.head_channels[i]
            body_channel = self.body_channels[i]

            # Check dead snakes all 0
            if self.snake_dones[:, i].sum() > 0:
                dead_envs = torch.cat([
                    self.envs[self.snake_dones[:, i], head_channel:head_channel+1:, :, :],
                    self.envs[self.snake_dones[:, i], body_channel:body_channel + 1, :, :]
                ], dim=1)
                dead_all_zeros = dead_envs.sum() == 0
                if not dead_all_zeros:
                    raise RuntimeError('Dead snake contains non-zero elements.')

            # Check living envs
            if (~self.snake_dones[:, i]).sum() > 0:
                living_envs = torch.cat([
                    self.envs[~self.snake_dones[:, i], 0:1, :, :],
                    self.envs[~self.snake_dones[:, i], head_channel:head_channel + 1, :, :],
                    self.envs[~self.snake_dones[:, i], body_channel:body_channel + 1, :, :]
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
        if done is None:
            done = self.env_dones

        done = done.view((done.shape[0]))

        t0 = time()
        # Create environments
        if done.sum() > 0:
            new_envs = self._create_envs(int(done.sum().item()))
            self.envs[done.byte(), :, :, :] = new_envs

        # Reset done counters
        self.snake_dones[done] = 0
        self.env_dones[done] = 0

        if self.verbose:
            print(f'Resetting {done.sum().item()} envs: {time() - t0}s')

        return self._observe()

    def _create_envs(self, num_envs: int):
        """Vectorised environment creation. Creates self.num_envs environments simultaneously."""
        if self.initial_snake_length != 3:
            raise NotImplementedError('Only initial snake length = 3 has been implemented.')

        envs = torch.zeros((num_envs, 1 + 2 * self.num_snakes, self.size, self.size), dtype=self.dtype, device=self.device)
        l = self.initial_snake_length-1

        for i in range(self.num_snakes):
            occupied_locations = envs.sum(dim=1, keepdim=True) > EPS
            # Expand this because you can't put a snake right next to another snake
            occupied_locations = (F.conv2d(
                occupied_locations.to(dtype=self.dtype),
                torch.ones((1, 1, 3, 3), dtype=self.dtype,device=self.device),
                padding=1
            ) > EPS).byte()

            available_locations = (envs.sum(dim=1, keepdim=True) < EPS) & ~occupied_locations

            # Remove boundaries
            available_locations[:, :, :l, :] = 0
            available_locations[:, :, :, :l] = 0
            available_locations[:, :, -l:, :] = 0
            available_locations[:, :, :, -l:] = 0

            # If there is no available locations for a snake raise an exception
            any_available_locations = available_locations.view(num_envs, -1).max(dim=1)[0].byte()
            if torch.any(~any_available_locations):
                raise RuntimeError('There is no available locations to create snake!')

            body_seed_indices = drop_duplicates(torch.nonzero(available_locations), 0)
            body_seeds = torch.sparse_coo_tensor(
                body_seed_indices.t(), torch.ones(len(body_seed_indices)), available_locations.shape,
                device=self.device, dtype=self.dtype
            )

            # Choose random starting directions
            random_directions = torch.randint(4, (num_envs,), device=self.device)
            random_directions_onehot = torch.empty((num_envs, 4), dtype=self.dtype, device=self.device)
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
            envs[:, self.body_channels[i], :, :] = bodies

            # Create num_heads at end of bodies
            snake_sizes = envs[:, self.body_channels[i], :].view(num_envs, -1).max(dim=1)[0]
            snake_size_mask = snake_sizes[:, None, None].expand((num_envs, self.size, self.size))
            envs[:, self.head_channels[i], :, :] = (bodies == snake_size_mask).to(dtype=self.dtype)

        # Add food
        food_addition = self._get_food_addition(envs)
        envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_addition

        return envs.round()
