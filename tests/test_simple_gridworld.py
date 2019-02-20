import unittest
import torch

from wurm.envs import SimpleGridworld
from wurm.utils import head
from config import FOOD_CHANNEL, HEAD_CHANNEL, DEFAULT_DEVICE


size = 7


class TestSimpleGridworld(unittest.TestCase):
    def test_basic_movement(self):
        env = SimpleGridworld(num_envs=1, size=size, start_location=(3, 3), manual_setup=True)
        env.envs[0, FOOD_CHANNEL, 1, 1] = 1
        env.envs[0, HEAD_CHANNEL, 3, 3] = 1

        actions = torch.Tensor([0, 1, 2, 3, 2, 1]).unsqueeze(1).long().to(DEFAULT_DEVICE)
        expected_head_positions = torch.Tensor([
            [4, 3],
            [4, 2],
            [3, 2],
            [3, 3],
            [2, 3],
            [2, 2]
        ])

        for i, a in enumerate(actions):

            observations, reward, done, info = env.step(a)

            head_position = torch.Tensor([
                head(env.envs)[0, 0].flatten().argmax() // size, head(env.envs)[0, 0].flatten().argmax() % size
            ])

            self.assertTrue(torch.equal(head_position, expected_head_positions[i]))

    def test_eat_food(self):
        env = SimpleGridworld(num_envs=1, size=size, start_location=(3, 3), manual_setup=True)
        env.envs[0, FOOD_CHANNEL, 1, 1] = 1
        env.envs[0, HEAD_CHANNEL, 2, 2] = 1

        actions = torch.Tensor([0, 2, 2, 1]).unsqueeze(1).long().to(DEFAULT_DEVICE)

        for i, a in enumerate(actions):

            observations, reward, done, info = env.step(a)

            if i == 4:
                self.assertEqual(reward.item(), 1)

        # Check for food respawn
        self.assertEqual(env.envs[0, FOOD_CHANNEL].sum().item(), 1)

    def test_edge_collision(self):
        env = SimpleGridworld(num_envs=1, size=size, start_location=(3, 3), manual_setup=True)
        env.envs[0, FOOD_CHANNEL, 1, 1] = 1
        env.envs[0, HEAD_CHANNEL, 3, 3] = 1

        actions = torch.Tensor([0, 0, 0, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE)

        for i, a in enumerate(actions):
            observations, reward, done, info = env.step(a)

            if i == 2:
                self.assertTrue(done.item())
                break
            else:
                self.assertFalse(done.item())

