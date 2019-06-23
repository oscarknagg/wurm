from time import time
from collections import namedtuple
import torch
from torch.nn import functional as F
from gym.envs.classic_control import rendering
import numpy as np
from PIL import Image

from config import DEFAULT_DEVICE, BODY_CHANNEL, EPS, HEAD_CHANNEL, FOOD_CHANNEL
from wurm._filters import ORIENTATION_DELTAS, NO_CHANGE_FILTER, LENGTH_3_SNAKES
from wurm.utils import determine_orientations, head, food, body, drop_duplicates


Spec = namedtuple('Spec', ['reward_threshold'])


class SingleSnake(object):
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
                 size: int,
                 max_timesteps: int = None,
                 initial_snake_length: int = 3,
                 on_death: str = 'restart',
                 observation_mode: str = 'one_channel',
                 device: str = DEFAULT_DEVICE,
                 manual_setup: bool = False,
                 verbose: int = 0,
                 render_args: dict = None):
        """Initialise the environments

        Args:
            num_envs:
            size:
            max_timesteps:
        """
        self.num_envs = num_envs
        self.size = size
        self.max_timesteps = max_timesteps
        self.initial_snake_length = initial_snake_length
        self.on_death = on_death
        self.observation_mode = observation_mode
        self.device = device
        self.verbose = verbose

        if render_args is None:
            self.render_args = {'num_rows': 1, 'num_cols': 1, 'size': 256}
        else:
            self.render_args = render_args

        self.envs = torch.zeros((num_envs, 3, size, size)).to(self.device).requires_grad_(False)
        self.t = 0

        if not manual_setup:
            # Create environments
            self.envs = self._create_envs(self.num_envs)
            self.envs.requires_grad_(False)

        self.done = torch.zeros((num_envs)).to(self.device).byte()

        self.viewer = None

        self.body_colour = torch.Tensor((0, 255 * 0.5, 0)).short().to(self.device)
        self.head_colour = torch.Tensor((0, 255, 0)).short().to(self.device)
        self.food_colour = torch.Tensor((255, 0, 0)).short().to(self.device)
        self.edge_colour = torch.Tensor((0, 0, 0)).short().to(self.device)

    def _get_rgb(self):
        # RGB image same as is displayed in .render()
        img = torch.ones_like(self.envs).short() * 255

        # Convert to BHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))

        body_locations = (body(self.envs) > EPS).squeeze(1)
        img[body_locations, :] = self.body_colour

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
            observation = self.envs.clone()
            return observation
        elif observation_mode == 'one_channel':
            observation = (self.envs[:, BODY_CHANNEL, :, :] > EPS).float() * 0.5
            observation += self.envs[:, HEAD_CHANNEL, :, :] * 0.5
            observation += self.envs[:, FOOD_CHANNEL, :, :] * 1.5
            # Add in -1 values to indicate edge of map
            observation[:, :1, :] = -1
            observation[:, :, :1] = -1
            observation[:, -1:, :] = -1
            observation[:, :, -1:] = -1
            return observation.unsqueeze(1)
        elif observation_mode == 'positions':
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
        elif observation_mode.startswith('partial_'):
            observation_size = int(self.observation_mode.split('_')[-1])
            observation_width = 2 * observation_size + 1

            # RGB image same as is displayed in .render()
            img = self._get_rgb()

            # Normalise to 0-1
            img = img.float() / 255

            # Pad envs so we ge tthe correct size observation even when the head of the snake
            # is close to the edge of the environment
            padding = [observation_size, observation_size, ] * 2
            padded_img = F.pad(img, padding)

            filter = torch.ones((1, 1, observation_width, observation_width)).to(self.device)
            head_area_indices = torch.nn.functional.conv2d(
                F.pad(self.envs.clone(), padding)[:, HEAD_CHANNEL:HEAD_CHANNEL + 1].clone(),
                filter,
                padding=observation_size
            ).round()

            observations = padded_img[
                head_area_indices.expand_as(padded_img).byte()
            ]
            observations = observations.view((self.num_envs, 3 * (observation_width ** 2))).clone()

            return observations
        else:
            raise Exception

    def step(self, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, dict):
        if actions.dtype not in (torch.short, torch.int, torch.long):
            raise TypeError('actions Tensor must be an integer type i.e. '
                            '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

        if actions.shape[0] != self.num_envs:
            raise RuntimeError('Must have the same number of actions as environments.')

        reward = torch.zeros((self.num_envs,)).float().to(self.device).requires_grad_(False)
        done = torch.zeros((self.num_envs,)).byte().to(self.device).byte().requires_grad_(False)
        info = dict()

        t0 = time()
        snake_sizes = self.envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :].view(self.num_envs, -1).max(dim=1)[0]

        orientations = determine_orientations(self.envs)
        if self.verbose > 0:
            print(f'\nOrientations: {time()-t0}s')

        t0 = time()
        # Check if any snakes are trying to move backwards and change
        # their direction/action to just continue forward
        # The test for this is if their orientation number {0, 1, 2, 3}
        # is the same as their action
        mask = orientations == actions
        actions.add_((mask * 2).long()).fmod_(4)

        # Create head position deltas
        head_deltas = F.conv2d(head(self.envs), ORIENTATION_DELTAS.to(self.device), padding=1)
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
        head_food_overlap = (head(self.envs) * food(self.envs)).view(self.num_envs, -1).sum(dim=-1)

        # Decay the body sizes by 1, hence moving the body, apply ReLu to keep above 0
        # Only do this for environments which haven't just eaten food
        body_decay_env_indices = ~head_food_overlap.byte()
        self.envs[body_decay_env_indices, BODY_CHANNEL:BODY_CHANNEL + 1, :, :] -= 1
        self.envs[body_decay_env_indices, BODY_CHANNEL:BODY_CHANNEL + 1, :, :] = \
            self.envs[body_decay_env_indices, BODY_CHANNEL:BODY_CHANNEL + 1, :, :].relu()

        # Check for hitting self
        self_collision = (head(self.envs) * body(self.envs)).view(self.num_envs, -1).sum(dim=-1) > EPS
        info.update({'self_collision': self_collision})
        done = done | self_collision

        # Create a new head position in the body channel
        # Make this head +1 greater if the snake has just eaten food
        self.envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :, :] += \
            head(self.envs) * (
                snake_sizes[:, None, None, None].expand((self.num_envs, 1, self.size, self.size)) +
                head_food_overlap[:, None, None, None].expand((self.num_envs, 1, self.size, self.size))
            )

        if self.verbose:
            print(f'Body movement: {time()-t0}')

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

    def render(self, mode: str = 'human'):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        # Get RBG Tensor BCHW
        img = self._get_rgb()

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
            output = np.zeros((self.size*num_rows, self.size*num_cols, 3))
            for i in range(num_rows):
                for j in range(num_cols):
                    output[
                        i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size, :
                    ] = np.transpose(img[i*num_cols + j], (1, 2, 0))

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

