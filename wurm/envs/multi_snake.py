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
from wurm.utils import determine_orientations, head, food, body, drop_duplicates, env_consistency


Spec = namedtuple('Spec', ['reward_threshold'])


class MultiSnakeEnvironments(object):
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

    def __init__(self,
                 num_envs: int,
                 num_snakes: int,
                 size: int,
                 initial_snake_length: int = 3,
                 on_death: str = 'restart',
                 observation_mode: str = 'one_channel',
                 device: str = DEFAULT_DEVICE,
                 manual_setup: bool = True,
                 verbose: int = 0):
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

        self.envs = torch.zeros((num_envs, 1 + 2 * num_snakes, size, size)).to(self.device).requires_grad_(False)
        self.head_channels = [1 + 2*i for i in range(self.num_snakes)]
        self.body_channels = [2 + 2*i for i in range(self.num_snakes)]
        self.t = 0
        self.viewer = 0

        if not manual_setup:
            # Create environments
            self.envs = self._create_envs(self.num_envs)
            self.envs.requires_grad_(False)

        self.done = torch.zeros(num_envs).to(self.device).byte()
        self.dead = torch.zeros(num_envs, num_snakes).to(self.device).byte()

        self.viewer = None

        self.body_colour = torch.Tensor((0, 255 * 0.5, 0)).short().to(self.device)
        self.head_colour = torch.Tensor((0, 255, 0)).short().to(self.device)
        self.food_colour = torch.Tensor((255, 0, 0)).short().to(self.device)
        self.edge_colour = torch.Tensor((0, 0, 0)).short().to(self.device)

    def _get_rgb(self):
        # RGB image same as is displayed in .render()
        img = torch.ones((self.num_envs, 3, self.size, self.size)).to(self.device).short() * 255

        # Convert to BHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))


        body_locations = ((self._bodies > EPS).squeeze(1).sum(dim=1) > EPS).byte()
        img[body_locations, :] = self.body_colour

        head_locations = ((self._heads > EPS).squeeze(1).sum(dim=1) > EPS).byte()
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

    def render(self):
        if self.num_envs != 1:
            raise RuntimeError('Rendering is only supported for a single environment at a time')

        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        # Get RBG Tensor BCHW
        img = self._get_rgb()

        # Convert to numpy, transpose to HWC and resize
        img = img.cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = np.array(Image.fromarray(img.astype(np.uint8)).resize((500, 500)))
        self.viewer.imshow(img)

        return self.viewer.isopen

    def _observe(self):
        if self.observation_mode == 'default':
            observation = self.envs.clone()
            # Add in -1 values to indicate edge of map
            observation[:, :, :1, :] = -1
            observation[:, :, :, :1] = -1
            observation[:, :, -1:, :] = -1
            observation[:, :, :, -1:] = -1
            return observation
        elif self.observation_mode == 'one_channel':
            observation = (self.envs[:, BODY_CHANNEL, :, :] > EPS).float() * 0.5
            observation += self.envs[:, HEAD_CHANNEL, :, :] * 0.5
            observation += self.envs[:, FOOD_CHANNEL, :, :] * 1.5
            # Add in -1 values to indicate edge of map
            observation[:, :1, :] = -1
            observation[:, :, :1] = -1
            observation[:, -1:, :] = -1
            observation[:, :, -1:] = -1
            return observation.unsqueeze(1)
        elif self.observation_mode == 'positions':
            observation = (self.envs[:, BODY_CHANNEL, :, :] > EPS).float() * 0.5
            observation += self.envs[:, HEAD_CHANNEL, :, :] * 0.5
            observation += self.envs[:, FOOD_CHANNEL, :, :] * 1.5
            head_idx = self.envs[:, HEAD_CHANNEL, :, :].view(self.num_envs, self.size ** 2).argmax(dim=-1)
            food_idx = self.envs[:, FOOD_CHANNEL, :, :].view(self.num_envs, self.size ** 2).argmax(dim=-1)
            size = torch.Tensor([self.size, ]*self.num_envs).long().to(self.device)
            observation = torch.stack([
                head_idx // size,
                head_idx % size,
                food_idx // size,
                food_idx % size
            ]).float().t()
            return observation
        elif self.observation_mode.startswith('partial_'):
            observation_size = int(self.observation_mode.split('_')[-1])
            observation_width = 2 * observation_size + 1

            # Pad envs so we ge tthe correct size observation even when the head of the snake
            # is close to the edge of the environment
            padding = [observation_size, observation_size, ] * 2
            padded_envs = F.pad(self.envs.clone(), padding)

            filter = torch.ones((1, 1, observation_width, observation_width)).to(self.device)
            head_area_indices = torch.nn.functional.conv2d(
                padded_envs[:, HEAD_CHANNEL:HEAD_CHANNEL + 1].clone(), filter, padding=observation_size
            ).round()

            # Add in -1 values to indicate edge of map
            padded_envs[:, :, :observation_size, :] = -1
            padded_envs[:, :, :, :observation_size] = -1
            padded_envs[:, :, -observation_size:, :] = -1
            padded_envs[:, :, :, -observation_size:] = -1

            observations = padded_envs[
                head_area_indices.expand_as(padded_envs).byte()
            ]
            observations = observations.view((self.num_envs, 3 * (observation_width ** 2))).clone()

            return observations
        else:
            raise Exception

    @property
    def _food(self):
        return self.envs[:, 0:1, :, :]

    @property
    def _heads(self):
        return self.envs[:, self.head_channels, :, :]

    @property
    def _bodies(self):
        return self.envs[:, self.body_channels, :, :]

    def step(self, actions: Dict[str, torch.Tensor]) -> Tuple[dict, dict, dict, dict]:
        if len(actions) != self.num_snakes:
            raise RuntimeError('Must have a Tensor of actions for each snake')

        for agent, act in actions.items():
            if act.dtype not in (torch.short, torch.int, torch.long):
                raise TypeError('actions Tensor must be an integer type i.e. '
                                '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

            if act.shape[0] != self.num_envs:
                raise RuntimeError('Must have the same number of actions as environments.')

        rewards = OrderedDict([
            (
                f'agent_{i}',
                torch.zeros((self.num_envs,)).float().to(self.device).requires_grad_(False)
            ) for i in range(self.num_snakes)
        ])
        dones = OrderedDict([
            (
                f'agent_{i}',
                torch.zeros((self.num_envs,)).float().to(self.device).byte().requires_grad_(False)
            ) for i in range(self.num_snakes)
        ])
        info = dict()

        snake_sizes = dict()
        for i, (agent, act) in enumerate(actions.items()):
            body_channel = self.body_channels[i]
            snake_sizes[agent] = self.envs[:, body_channel:body_channel + 1, :].view(self.num_envs, -1).max(dim=1)[0]

        # Check orientations and move head positions of all snakes
        for i, (agent, act) in enumerate(actions.items()):
            # The sub-environment of just one agent
            head_channel = self.head_channels[i]
            body_channel = self.body_channels[i]
            _env = self.envs[:, [0, head_channel, body_channel], :, :]

            orientations = determine_orientations(_env)

            # Check if this snake is trying to move backwards and change
            # it's direction/action to just continue forward
            # The test for this is if their orientation number {0, 1, 2, 3}
            # is the same as their action
            mask = orientations == act
            act.add_((mask * 2).long()).fmod_(4)

            # Create head position deltas
            head_deltas = F.conv2d(head(_env), ORIENTATION_FILTERS.to(self.device), padding=1)
            # Select the head position delta corresponding to the correct action
            actions_onehot = torch.Tensor(self.num_envs, 4).float().to(self.device)
            actions_onehot.zero_()
            actions_onehot.scatter_(1, act.unsqueeze(-1), 1)
            head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, actions_onehot]).unsqueeze(1)

            # Move head position by applying delta
            self.envs[:, head_channel:head_channel + 1, :, :].add_(head_deltas).round_()

        # Decay bodies of all snakes that haven't eaten food
        food_consumption = dict()
        for i, (agent, act) in enumerate(actions.items()):
            head_channel = self.head_channels[i]
            body_channel = self.body_channels[i]
            _env = self.envs[:, [0, head_channel, body_channel], :, :]

            head_food_overlap = (head(_env) * food(_env)).view(self.num_envs, -1).sum(dim=-1)
            food_consumption[agent] = head_food_overlap

            # Decay the body sizes by 1, hence moving the body, apply ReLu to keep above 0
            # Only do this for environments which haven't just eaten food
            body_decay_env_indices = ~head_food_overlap.byte()
            self.envs[body_decay_env_indices, body_channel:body_channel + 1, :, :] -= 1
            self.envs[body_decay_env_indices, body_channel:body_channel + 1, :, :] = \
                self.envs[body_decay_env_indices, body_channel:body_channel + 1, :, :].relu()

        for i, (agent, act) in enumerate(actions.items()):
            # Check if any snakes have collided with themselves or any other snakes
            head_channel = self.head_channels[i]
            body_channel = self.body_channels[i]
            _env = self.envs[:, [0, head_channel, body_channel], :, :]

            # Collision with body of any snake
            body_collision = (head(_env) * self._bodies).view(self.num_envs, -1).sum(dim=-1) > EPS
            # Collision with head of other snake
            other_snakes = torch.ones(self.num_snakes).byte().to(self.device)
            other_snakes[i] = 0
            other_heads = self._heads[:, other_snakes, :, :]
            head_collision = (head(_env) * other_heads).view(self.num_envs, -1).sum(dim=-1) > EPS
            snake_collision = body_collision | head_collision
            info.update({f'self_collision_{i}': snake_collision})
            dones[agent] = dones[agent] | snake_collision

            # Create a new head position in the body channel
            # Make this head +1 greater if the snake has just eaten food
            self.envs[:, body_channel:body_channel + 1, :, :] += \
                head(_env) * (
                    snake_sizes[agent][:, None, None, None].expand((self.num_envs, 1, self.size, self.size)) +
                    food_consumption[agent][:, None, None, None].expand((self.num_envs, 1, self.size, self.size))
                )

        for i, (agent, act) in enumerate(actions.items()):
            # Remove food and give reward
            # `food_removal` is 0 except where a snake head is at the same location as food where it is -1
            head_channel = self.head_channels[i]
            body_channel = self.body_channels[i]
            _env = self.envs[:, [0, head_channel, body_channel], :, :]

            food_removal = head(_env) * food(_env) * -1
            rewards[agent].sub_(food_removal.view(self.num_envs, -1).sum(dim=-1).float())
            self.envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_removal

        # Add new food if necessary.
        food_addition_env_indices = (food(self.envs).view(self.num_envs, -1).sum(dim=-1) < EPS)
        if food_addition_env_indices.sum().item() > 0:
            add_food_envs = self.envs[food_addition_env_indices, :, :, :]
            food_addition = self._get_food_addition(add_food_envs)
            self.envs[food_addition_env_indices, FOOD_CHANNEL:FOOD_CHANNEL+1, :, :] += food_addition

        for i, (agent, act) in enumerate(actions.items()):
            # Check for boundary, Done by performing a convolution with no padding
            # If the head is at the edge then it will be cut off and the sum of the head
            # channel will be 0
            head_channel = self.head_channels[i]
            body_channel = self.body_channels[i]
            _env = self.envs[:, [0, head_channel, body_channel], :, :]
            edge_collision = F.conv2d(
                head(_env),
                NO_CHANGE_FILTER.to(self.device),
            ).view(self.num_envs, -1).sum(dim=-1) < EPS
            dones[agent] = dones[agent] | edge_collision
            info.update({f'edge_collision_{i}': edge_collision})

        for i, (agent, act) in enumerate(actions.items()):
            # Remove any snakes that are dead
            # self._bodies (num_envs, num_snakes, size, size)
            self._bodies[dones[agent], i, 0, 0] = 0
            self._heads[dones[agent], i, 0, 0] = 0

            # TODO:
            # Keep track of which snakes are already dead not just which have died
            # in the current step

        # Apply rounding to stop numerical errors accumulating
        self.envs.round_()

        # Environment is finished if all snake are dead
        dones['__all__'] = torch.ones((self.num_envs,)).float().to(self.device).byte().requires_grad_(False)
        for agent, act in actions.items():
            dones['__all__'] = dones['__all__'] & dones[agent]

        self.done = dones['__all__']

        return dict(), rewards, dones, info

    def check_consistency(self):
        """Runs multiple checks for environment consistency and throws an exception if any fail"""
        n = self.num_envs

        for i in range(self.num_snakes):
            head_channel = self.head_channels[i]
            body_channel = self.body_channels[i]
            _env = self.envs[:, [0, head_channel, body_channel], :, :]
            env_consistency(_env)

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

        return self._observe()

    def _create_envs(self, num_envs: int):
        """Vectorised environment creation. Creates self.num_envs environments simultaneously."""
        if self.size <= 8:
            raise NotImplementedError('Cannot make an env this small without making this code more clever')

        if self.initial_snake_length != 3:
            raise NotImplementedError('Only initial snake length = 3 has been implemented.')

        envs = torch.zeros((num_envs, 3, self.size, self.size)).to(self.device)

        # Create random locations to seed bodies
        body_seed_indices = torch.stack([
            torch.arange(num_envs),
            torch.zeros((num_envs,)).long(),
            torch.randint(1 + self.initial_snake_length, self.size - (1 + self.initial_snake_length), size=(num_envs,)),
            torch.randint(1 + self.initial_snake_length, self.size - (1 + self.initial_snake_length), size=(num_envs,))
        ]).to(self.device)
        body_seeds = torch.sparse_coo_tensor(
            body_seed_indices, torch.ones(num_envs), (num_envs, 1, self.size, self.size), device=self.device
        )

        # Choose random starting directions
        random_directions = torch.randint(4, (num_envs,)).to(self.device)
        random_directions_onehot = torch.Tensor(num_envs, 4).float().to(self.device)
        random_directions_onehot.zero_()
        random_directions_onehot.scatter_(1, random_directions.unsqueeze(-1), 1)

        # Create bodies
        bodies = torch.einsum('bchw,bc->bhw', [
            F.conv2d(body_seeds.to_dense(), LENGTH_3_SNAKES.to(self.device), padding=1),
            random_directions_onehot
        ]).unsqueeze(1)
        envs[:, BODY_CHANNEL:BODY_CHANNEL+1, :, :] = bodies

        # Create num_heads at end of bodies
        snake_sizes = envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :].view(num_envs, -1).max(dim=1)[0]
        snake_size_mask = snake_sizes[:, None, None, None].expand((num_envs, 1, self.size, self.size))
        envs[:, HEAD_CHANNEL:HEAD_CHANNEL + 1, :, :] = (bodies == snake_size_mask).float()

        # Add food
        food_addition = self._get_food_addition(envs)
        envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_addition

        return envs.round()
