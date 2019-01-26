import torch
import torch.nn.functional as F

from config import FOOD_CHANNEL, HEAD_CHANNEL, BODY_CHANNEL
from wurm._filters import ORIENTATION_FILTERS

"""
Each of the following functions takes a batch of envs and returns the channel corresponding to either the food, heads or
bodies of each env.

Return shape should be (num_envs, 1, size, size)
"""


def food(envs: torch.Tensor) -> torch.Tensor:
    return envs[:, FOOD_CHANNEL:FOOD_CHANNEL+1]


def head(envs: torch.Tensor) -> torch.Tensor:
    return envs[:, HEAD_CHANNEL:HEAD_CHANNEL+1]


def body(envs: torch.Tensor) -> torch.Tensor:
    return envs[:, BODY_CHANNEL:BODY_CHANNEL+1]


def determine_orientations(envs: torch.Tensor) -> torch.Tensor:
    """Returns a batch of snake orientations from a batch of envs.

    Args:
        envs: Batch of environments

    Returns:
        orientations: A 1D Tensor with the same length as the number of envs. Each element should be a
        number {0, 1, 2, 3} representation the orientation of the head of the snake in each environment.
    """
    # Generate "neck" channel where the head is +1, the next piece of the snake
    # is -1 and the rest are 0
    n = envs.shape[0]
    bodies = envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :]
    snake_sizes = bodies.view(n, -1).max(dim=1)[0]
    snake_sizes.sub_(2)
    shift = snake_sizes[:, None, None, None].expand_as(bodies)
    necks = F.relu(bodies - shift)
    necks.sub_(1.5 * (torch.ones_like(necks) * (necks > 0).float()))
    necks.mul_(2)

    # Convolve with 4 predetermined filters one of which will be more activated
    # because it lines up with the orientation of the snake
    responses = F.conv2d(necks, ORIENTATION_FILTERS, padding=1)

    # Find which filter
    responses = responses.view(n, 4, -1)
    orientations = responses.max(dim=-1)[0].argmax(dim=-1)

    return orientations


def get_test_env(size: int, orientation: str = 'up') -> torch.Tensor:
    """Gets a predetermined single-snake environment with the snake in a specified orientation"""
    env = torch.zeros((1, 3, size, size))
    if orientation == 'up':
        env[0, BODY_CHANNEL, 3, 3] = 1
        env[0, BODY_CHANNEL, 3, 4] = 2
        env[0, BODY_CHANNEL, 4, 4] = 3
        env[0, BODY_CHANNEL, 5, 4] = 4

        env[0, HEAD_CHANNEL, 5, 4] = 1

        env[0, FOOD_CHANNEL, 6, 6] = 1
    elif orientation == 'right':
        env[0, BODY_CHANNEL, 3, 3] = 1
        env[0, BODY_CHANNEL, 3, 4] = 2
        env[0, BODY_CHANNEL, 4, 4] = 3
        env[0, BODY_CHANNEL, 4, 5] = 4

        env[0, HEAD_CHANNEL, 4, 5] = 1

        env[0, FOOD_CHANNEL, 6, 9] = 1
    elif orientation == 'down':
        env[0, BODY_CHANNEL, 8, 8] = 1
        env[0, BODY_CHANNEL, 7, 8] = 2
        env[0, BODY_CHANNEL, 6, 8] = 3
        env[0, BODY_CHANNEL, 5, 8] = 4

        env[0, HEAD_CHANNEL, 5, 8] = 1

        env[0, FOOD_CHANNEL, 7, 2] = 1
    elif orientation == 'left':
        env[0, BODY_CHANNEL, 8, 7] = 1
        env[0, BODY_CHANNEL, 7, 7] = 2
        env[0, BODY_CHANNEL, 6, 7] = 3
        env[0, BODY_CHANNEL, 6, 6] = 4

        env[0, HEAD_CHANNEL, 6, 6] = 1

        env[0, FOOD_CHANNEL, 1, 2] = 1
    else:
        raise Exception

    return env


def env_consistency(envs: torch.Tensor):
    """Runs multiple checks for environment consistency and throws an exception if any fail"""
    n = envs.shape[0]

    one_head_per_snake = torch.all(head(envs).view(n, -1).sum(dim=-1) == 1)
    if not one_head_per_snake:
        raise RuntimeError('An environment has multiple heads for a single snake.')

    # Head is at end of body

    # Body is in decreasing order
