import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import gc
import git
from pprint import pformat
import json

from torch import nn

from config import FOOD_CHANNEL, HEAD_CHANNEL, BODY_CHANNEL
from wurm._filters import ORIENTATION_DELTAS
from wurm.core import MultiagentVecEnv
from wurm.interaction import InteractionHandler

"""
Each of the following functions takes a batch of envs and returns the channel corresponding to either the food, num_heads or
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
    necks.sub_(1.5 * (torch.ones_like(necks) * (necks > 0).to(dtype=envs.dtype)))
    necks.mul_(2)

    # Convolve with 4 predetermined filters one of which will be more activated
    # because it lines up with the orientation of the snake
    responses = F.conv2d(necks, ORIENTATION_DELTAS.to(device=envs.device, dtype=envs.dtype), padding=1)

    # Find which filter
    responses = responses.view(n, 4, -1)
    orientations = responses.max(dim=-1)[0].argmax(dim=-1)

    return orientations


def pad_to_square(image_batch: torch.Tensor, padding_value: float = 0) -> torch.Tensor:
    _, _, h, w = image_batch.size()
    if h != w:
        side_difference = h - w if h > w else w - h
        to_square_padding = [0, side_difference]
        to_square_padding = to_square_padding if h > w else [0, 0] + to_square_padding
        image_batch = F.pad(image_batch, to_square_padding, value=padding_value)

    return image_batch


def unpad_from_square(image_batch: torch.Tensor, original_h: int, original_w: int) -> torch.Tensor:
    """Reverses pad_to_square()"""
    h, w = original_h, original_w

    if h == w:
        return image_batch

    side_difference = h - w if h > w else w - h
    if h > w:
        image_batch = image_batch[:, :, :, :-side_difference]
    else:
        image_batch = image_batch[:, :, :-side_difference, :]

    return image_batch


def snake_consistency(envs: torch.Tensor):
    """Checks for consistency of a 3 channel single-snake env."""
    n = envs.shape[0]
    if n == 0:
        return

    food_is_zero = food(envs) == 0
    food_is_one = food(envs) == 1
    if not torch.all(food_is_zero | food_is_one):
        offending_envs = (~(food_is_zero | food_is_one)).view(n, -1).any(dim=-1)
        print(envs[offending_envs][0].long())
        print(offending_envs.nonzero().flatten())
        raise RuntimeError('An environment has an invalid food pixel')

    one_head_per_snake = torch.all(head(envs).view(n, -1).sum(dim=-1) == 1)
    if not one_head_per_snake:
        print('Snake head sums: ')
        print(head(envs).view(n, -1).sum(dim=-1))
        raise RuntimeError('An environment has multiple num_heads for a single snake.')

    # Environment contains a snake
    envs_contain_snake = body(envs).view(n, -1).sum(dim=-1) > 0
    if not torch.all(envs_contain_snake):
        raise RuntimeError(f'{(~envs_contain_snake).sum()} environments don\'t contain a snake.')

    # Head is at end of body
    body_value_at_head_locations = (head(envs) * body(envs))
    body_sizes = body(envs).view(n, -1).max(dim=1)[0]
    head_is_at_end_of_body = torch.equal(body_sizes, body_value_at_head_locations.view(n, -1).sum(dim=-1))
    if not head_is_at_end_of_body:
        raise RuntimeError('An environment has a snake with it\'s head not at the end of the body.')

    # Sum of body channel is a triangular number
    # This checks that the body contains elements like {1, 2, 3}, {1, 2, 3, 4}, range(1, n)
    body_totals = body(envs).view(n, -1).sum(dim=-1)
    estimated_body_sizes = (torch.sqrt(8 * body_totals + 1) - 1) / 2
    consistent_body_size = torch.equal(estimated_body_sizes, body_sizes)
    if not consistent_body_size:
        print(envs[estimated_body_sizes != body_sizes][0])
        print(torch.nonzero(estimated_body_sizes != body_sizes))
        raise RuntimeError('An environment has a body with inconsistent values i.e. not range(n)')

    # No snakes of length less than 3 (size 6)
    if not torch.all(body_totals >= 6):
        raise RuntimeError('A snake has size of less than 3.')

    # No food and head overlap
    head_food_overlap = (head(envs) * food(envs)).view(n, -1).sum(dim=-1)
    if not torch.all(head_food_overlap == 0):
        print(torch.nonzero(head_food_overlap))
        print(envs[torch.nonzero(head_food_overlap)[0]].long())
        raise RuntimeError(f'A food and head pixel is overlapping in {int(head_food_overlap.sum().item())} env(s).')


def env_consistency(envs: torch.Tensor):
    """Runs multiple checks for environment consistency and throws an exception if any fail"""
    snake_consistency(envs)

    n = envs.shape[0]
    if n == 0:
        return

    # Environment contains one food instance
    contains_one_food = torch.all(food(envs).view(n, -1).sum(dim=-1) == 1)
    if not contains_one_food:
        raise RuntimeError('An environment doesn\'t contain exactly one food instance')


def unique1d(tensor: torch.Tensor, return_index: bool = False):
    """Port of np.unique to PyTorch with `return_index` functionality"""
    assert len(tensor.shape) == 1

    optional_indices = return_index

    if optional_indices:
        perm = tensor.argsort()
        aux = tensor[perm]
    else:
        tensor.sort_()
        aux = tensor

    mask = torch.zeros(aux.shape)
    mask[:1] = 1
    mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask.byte()],)
    if return_index:
        ret += (perm[mask.byte()],)

    return ret


def drop_duplicates(tensor: torch.Tensor, column: int, random: bool = True):
    """Equivalent of pandas.drop_duplicates

    Takes a 2D tensor and returns the same tensor without any rows that contain a duplicate value in a
    particular column.

    Args:
        tensor: A 2D tensor
        column: The column (i.e. 2nd index) of the tensor to place a unique constraint on
        random: If `False` returns the first occurrence of rows with duplicate values. If `True`
            returns a random row from those that contain a duplicated value.

    Returns:
        unique: Tensor with no duplicate values in a particular column
    """
    if not len(tensor.shape) == 2:
        raise RuntimeError('Input must be a 2D tensor.')

    if random:
        tensor = tensor[torch.randperm(len(tensor))]

    uniq = tensor[:, column]

    indices = unique1d(uniq, return_index=True)[1]

    unique = tensor[indices.long()]

    return unique


class ExponentialMovingAverageTracker(object):
    def __init__(self, alpha: float):
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.smoothed_values = {}

    def __call__(self, **kwargs):
        """Takes in raw values and outputs smoothed values"""
        for k, v in kwargs.items():
            if k not in self.smoothed_values.keys():
                self.smoothed_values[k] = v
            else:
                self.smoothed_values[k] = self.alpha * v + (1 - self.alpha) * self.smoothed_values[k]

        return self.smoothed_values

    def __getitem__(self, item):
        return self.smoothed_values[item]


def print_alive_tensors():
    """Prints the tensors that are currently alive in memory.

    Useful for debugging memory leaks.
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def autograd_graph(tensor: torch.Tensor) -> Tuple[
            List[torch.autograd.Function],
            List[Tuple[torch.autograd.Function, torch.autograd.Function]]
        ]:
    """Recursively retrieves the autograd graph for a particular tensor.
    # Arguments
        tensor: The Tensor to retrieve the autograd graph for
    # Returns
        nodes: List of torch.autograd.Functions that are the nodes of the autograd graph
        edges: List of (Function, Function) tuples that are the edges between the nodes of the autograd graph
    """
    nodes, edges = list(), list()

    def _add_nodes(tensor):
        if tensor not in nodes:
            nodes.append(tensor)

            if hasattr(tensor, 'next_functions'):
                for f in tensor.next_functions:
                    if f[0] is not None:
                        edges.append((f[0], tensor))
                        _add_nodes(f[0])

            if hasattr(tensor, 'saved_tensors'):
                for t in tensor.saved_tensors:
                    edges.append((t, tensor))
                    _add_nodes(t)

    _add_nodes(tensor.grad_fn)

    return nodes, edges


def rotate_image_batch(img: torch.Tensor, degree: int = 0) -> torch.Tensor:
    n, c, h, w = img.size()

    if degree == 0:
        rot = img
    elif degree == 90:
        rot = img.transpose(3, 2).flip(3)
    elif degree == 180:
        rot = img.flip(2).flip(3)
    else:
        rot = img.transpose(3, 2).flip(2)

    return rot


def get_comment(args) -> str:
    """Gets a verbose comment to add at the top of a CSV log file.

    Args:
        args: Input command line arguments.
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    comment = f'Git commit: {sha}\n'
    comment += f'Args: {json.dumps(args.__dict__)}\n'
    comment += 'Prettier args:\n'
    comment += pformat(args.__dict__)
    return comment


def stack_dict_of_tensors(d: Dict[str, torch.Tensor], take_every: int = 1) -> torch.Tensor:
    stacked = []
    for i, (k, v) in enumerate(d.items()):
        if i % take_every == 0:
            stacked.append(v)

    stacked = torch.stack(stacked).view(-1, 1)
    return stacked


class WarmStart:
    """Runs env for some steps before training starts."""
    def __init__(self, env: MultiagentVecEnv, models: List[nn.Module], num_steps: int, interaction_handler: InteractionHandler):
        super(WarmStart, self).__init__()
        self.env = env
        self.models = models
        self.num_steps = num_steps
        self.interaction_handler = interaction_handler

    def warm_start(self):
        # Run all agents for warm_start steps before training
        observations = self.env.reset()
        hidden_states = {f'agent_{i}': torch.zeros(
            (self.env.num_envs, 64), device=self.env.device) for i in range(self.env.num_agents)}
        cell_states = {f'agent_{i}': torch.zeros(
            (self.env.num_envs, 64), device=self.env.device) for i in range(self.env.num_agents)}

        for i in range(self.num_steps):
            interaction = self.interaction_handler.interact(observations, hidden_states, cell_states)

            observations, reward, done, info = self.env.step(interaction.actions)

            self.env.reset(done['__all__'])
            self.env.check_consistency()

            hidden_states = {k: v.detach() for k, v in hidden_states.items()}
            cell_states = {k: v.detach() for k, v in cell_states.items()}

        return observations, hidden_states, cell_states
