import unittest
import pytest
import torch
from time import sleep

from wurm.envs import LaserTag


render_envs = True
size = 9


def get_test_env(num_envs=2):
    # Same as small2 map from the Deepmind paper
    env = LaserTag(num_envs, 2, size)
    for i in range(num_envs):
        env.agents[2*i, :, 1, 1] = 1
        env.agents[2*i + 1, :, 7, 7] = 1
        env.orientations = torch.tensor([0, 2]*num_envs, dtype=torch.long, device=env.device, requires_grad=False)

    env.pathing[:, :, 3, 3] = 1
    env.pathing[:, :, 3, 5] = 1
    env.pathing[:, :, 4, 2] = 1
    env.pathing[:, :, 4, 3] = 1
    env.pathing[:, :, 4, 5] = 1
    env.pathing[:, :, 4, 6] = 1
    env.pathing[:, :, 5, 3] = 1
    env.pathing[:, :, 5, 5] = 1

    return env


class TestLaserTag(unittest.TestCase):
    def test_basic_movement(self):
        pass

    def test_render(self):
        env = get_test_env()

        if render_envs:
            env.render()
            sleep(5)
