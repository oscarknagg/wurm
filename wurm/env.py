import torch
import torch.nn.functional as F
from typing import List

from config import FOOD_CHANNEL, HEAD_CHANNEL, BODY_CHANNEL, DEFAULT_DEVICE
from wurm.utils import food, head, body, determine_orientations
from wurm._filters import *


class SingleSnakeEnvironments(object):
    """Many environments containing a single snake as a batched image tensor.

    Each environment has 3 channels. Each index has the following meaning
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
    def __init__(self,
                 num_envs: int,
                 size: int,
                 max_timesteps: int = None,
                 initial_snake_length: int = 4,
                 on_death: str = 'restart',
                 device: str = DEFAULT_DEVICE):
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
        self.device = device

        self.envs = torch.zeros((num_envs, 3, size, size)).to(self.device)
        self.t = 0

    def step(self, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, List[dict]):
        n = self.envs.shape[0]

        if actions.dtype not in (torch.short, torch.int, torch.long):
            raise TypeError('actions Tensor must be an integer type i.e. '
                            '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

        # assert n == actions.shape[0]

        reward = torch.zeros((n,)).long().to(self.device)
        done = torch.zeros((n,)).byte().to(self.device)
        info = [dict(), ] * n

        snake_sizes = self.envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :].view(n, -1).max(dim=1)[0]

        orientations = determine_orientations(self.envs)

        # Check if any snakes are trying to move backwards and change
        # their direction/action to just continue forward
        # The test for this is if their orientation number {0, 1, 2, 3}
        # is the same as their action
        mask = orientations == actions
        actions.add_((mask * 2).long()).fmod_(4)

        # Create head position deltas
        head_deltas = F.conv2d(head(self.envs), ORIENTATION_FILTERS, padding=1)

        # Mask the head-movement deltas with direction action
        head_deltas = torch.index_select(head_deltas, 1, actions.long())

        # Move head position by applying delta
        self.envs[:, HEAD_CHANNEL:HEAD_CHANNEL + 1, :, :].add_(head_deltas)

        # Check for hitting self
        hit_self = (head(self.envs) * body(self.envs)).view(n, -1).sum(dim=-1) > 0
        done = torch.clamp(done + hit_self, 0, 1)

        ################
        # Apply update #
        ################

        # Decay the body sizes by 1, hence moving the body
        body_movement = torch.zeros_like(self.envs)
        body_movement[:, BODY_CHANNEL, :, :] = -1
        self.envs = F.relu(self.envs + body_movement)
        # Create a new head position in the body channel
        self.envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :, :] += \
            head(self.envs) * snake_sizes[:, None, None, None].expand((n, 1, self.size, self.size))
        # Apply food growth i.e. +1 to all snake locations
        self.envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :, :] += \
            ((head(self.envs) * food(self.envs)).sum() > 0) * body(self.envs).clamp(0, 1)

        # Remove food and give reward
        # This Tensor is 0 except where a snake head is at the same location as food
        food_removal = head(self.envs) * food(self.envs) * -1
        reward.sub_(food_removal.view(n, -1).sum(dim=-1).long())
        self.envs[:, FOOD_CHANNEL:FOOD_CHANNEL + 1, :, :] += food_removal

        # TODO: Add new food if necessary
        pass  # Add a random food location multiplied by ((head * food).sum() > 0)

        # Check for boundary, Done by performing a convolution with no padding
        # If the head is at the edge then it will be cut off and the sum of the head
        # channel will be 0
        head_at_edge = F.conv2d(
            head(self.envs),
            NO_CHANGE_FILTER,
        ).view(n, -1).sum(dim=-1) == 0
        done = torch.clamp(done + head_at_edge, 0, 1)

        return self.envs, reward, done, info

    def reset(self):
        raise NotImplementedError



