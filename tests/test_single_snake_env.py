import unittest
import torch
import matplotlib.pyplot as plt
from time import time

from wurm.utils import get_test_env, head, body, food, env_consistency
from wurm.env import SingleSnakeEnvironments
from wurm.vis import plot_envs
from config import DEFAULT_DEVICE


visualise = False
size = 12


class TestSingleSnakeEnv(unittest.TestCase):
    def test_multiple_envs(self):
        num_envs = 1024
        num_steps = 50
        env = SingleSnakeEnvironments(num_envs=num_envs, size=size)
        actions = torch.randint(4, size=(num_steps, num_envs)).long().to(DEFAULT_DEVICE)

        t0 = time()
        for i, a in enumerate(actions):
            observations, reward, done, info = env.step(a)
            env.reset(done)
            env_consistency(env.envs)
            print()

        t = time() - t0
        print(f'Ran {num_envs*num_steps} actions in {t}s = {num_envs*num_steps/t} actions/s')

    def test_setup(self):
        env = SingleSnakeEnvironments(num_envs=100, size=size)
        env_consistency(env.envs)

    def test_reset(self):
        env = SingleSnakeEnvironments(num_envs=1, size=size)
        env_consistency(env.envs)
        env.reset(torch.Tensor([1]))
        env_consistency(env.envs)

    def test_basic_movement(self):
        env = SingleSnakeEnvironments(num_envs=1, size=size, manual_setup=True)
        env.envs = get_test_env(size, 'up').to(DEFAULT_DEVICE)
        actions = torch.Tensor([0, 0, 3, 0, 0, 1]).unsqueeze(1).long().to(DEFAULT_DEVICE)
        expected_head_positions = torch.Tensor([
            [6, 4],
            [7, 4],
            [7, 5],
            [8, 5],
            [9, 5],
            [9, 4]
        ])

        for i, a in enumerate(actions):
            if visualise:
                plot_envs(env.envs)
                plt.show()

            observations, reward, done, info = env.step(a)

            head_position = torch.Tensor([
                head(env.envs)[0, 0].flatten().argmax() // size, head(env.envs)[0, 0].flatten().argmax() % size
            ])

            self.assertTrue(torch.equal(head_position, expected_head_positions[i]))

        if visualise:
            plot_envs(env.envs)
            plt.show()

    def test_eat_food(self):
        env = SingleSnakeEnvironments(num_envs=1, size=size, manual_setup=True)
        env.envs = get_test_env(size, 'up').to(DEFAULT_DEVICE)
        actions = torch.Tensor([0, 3, 3, 0, 0, 1]).unsqueeze(1).long().to(DEFAULT_DEVICE)

        initial_size = body(env.envs).max()

        for i, a in enumerate(actions):
            if visualise:
                plot_envs(env.envs)
                plt.show()

            observations, reward, done, info = env.step(a)

        final_size = body(env.envs).max()
        self.assertGreater(final_size, initial_size)

        if visualise:
            plot_envs(env.envs)
            plt.show()

    def test_hit_boundary(self):
        env = SingleSnakeEnvironments(num_envs=1, size=size, manual_setup=True)
        env.envs = get_test_env(size, 'up').to(DEFAULT_DEVICE)
        actions = torch.Tensor([1, ] * 10).unsqueeze(1).long().to(DEFAULT_DEVICE)

        hit_boundary = False

        for i, a in enumerate(actions):
            if visualise:
                plot_envs(env.envs)
                plt.show()

            observations, reward, done, info = env.step(a)

            if torch.any(done):
                hit_boundary = True
                break

        self.assertTrue(hit_boundary)

        if visualise:
            plot_envs(env.envs)
            plt.show()

    def test_hit_self(self):
        env = SingleSnakeEnvironments(num_envs=1, size=size, manual_setup=True)
        env.envs = get_test_env(size, 'up').to(DEFAULT_DEVICE)
        actions = torch.Tensor([0, 3, 3, 2, 1, 0, 0, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE)

        hit_self = False

        for i, a in enumerate(actions):
            if visualise:
                plot_envs(env.envs)
                plt.show()

            observations, reward, done, info = env.step(a)

            if torch.any(done):
                hit_self = True
                break

        self.assertTrue(hit_self)

        # Food is created agin after being eaten
        num_food = food(env.envs).sum()
        self.assertEqual(num_food, 1)

        if visualise:
            plot_envs(env.envs)
            plt.show()

    def test_cannot_move_backwards(self):
        env = SingleSnakeEnvironments(num_envs=1, size=size, manual_setup=True)
        env.envs = get_test_env(size, 'up').to(DEFAULT_DEVICE)
        actions = torch.Tensor([2, 2, 2, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE)
        expected_head_positions = torch.Tensor([
            [6, 4],
            [7, 4],
            [8, 4],
            [8, 5],
        ])

        for i, a in enumerate(actions):
            if visualise:
                plot_envs(env.envs)
                plt.show()

            observations, reward, done, info = env.step(a)

            head_position = torch.Tensor([
                head(env.envs)[0, 0].flatten().argmax() // size, head(env.envs)[0, 0].flatten().argmax() % size
            ])

            self.assertTrue(torch.equal(head_position, expected_head_positions[i]))

        if visualise:
            plot_envs(env.envs)
            plt.show()
