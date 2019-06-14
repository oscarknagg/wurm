import torch
import torch.nn.functional as F
from collections import Iterable, OrderedDict
from typing import List, Tuple
from pprint import pformat
import numpy as np
import csv
import gc
import os
import io

from config import FOOD_CHANNEL, HEAD_CHANNEL, BODY_CHANNEL
from wurm._filters import ORIENTATION_FILTERS


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
    responses = F.conv2d(necks, ORIENTATION_FILTERS.to(device=envs.device, dtype=envs.dtype), padding=1)

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


class CSVLogger(object):
    """Stream results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    Args:
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing training). False: overwrite existing file.
        header_comment: A possibly multi line string to put at the top of the CSV file
        """

    def __init__(self, filename: str, separator: str = ',', append: bool = False, header_comment: str = None):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.header_comment = header_comment
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = ''
        self._open_args = {'newline': '\n'}

        # Make directory
        os.makedirs(os.path.split(filename)[0], exist_ok=True)

        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'

        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def write(self, logs: dict):
        def handle_value(k):
            is_zero_dim_tensor = isinstance(k, torch.Tensor) and k.ndimension() == 0
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray and not is_zero_dim_tensor:
                return '"[%s]"' % (', '.join(map(str, k)))
            elif is_zero_dim_tensor:
                return k.item()
            else:
                return k

        def comment_out(s, comment='#'):
            return comment + s.replace('\n', f'\n{comment}')

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:
            # Write the initial comment
            if self.append_header:
                print(comment_out(self.header_comment), file=self.csv_file)

            # Write the column names
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = self.keys
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict()
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class PrintLogger(object):
    def __init__(self):
        pass

    def write(self, logs: dict):
        print(logs)


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
