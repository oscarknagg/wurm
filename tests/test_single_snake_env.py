import unittest
import torch
import matplotlib.pyplot as plt
from time import time
from typing import Union, List

from wurm.utils import head, body, food, env_consistency
from wurm.envs import SingleSnake
from config import DEFAULT_DEVICE, BODY_CHANNEL, HEAD_CHANNEL, FOOD_CHANNEL


visualise = False
size = 12


def plot_envs(
        envs: torch.Tensor,
        env_idx: Union[int, List[int]] = 0,
        mode: str = 'single'):
    """Plots a single environment from a batch of environments"""
    size = envs.shape[-1]

    if mode == 'single':
        img = (envs[env_idx, BODY_CHANNEL, :, :].cpu().numpy() > EPS) * 0.5
        img += envs[env_idx, HEAD_CHANNEL, :, :].cpu().numpy() * 0.5
        img += envs[env_idx, FOOD_CHANNEL, :, :].cpu().numpy() * 1.5
        plt.imshow(img, vmin=0, vmax=1.5)
        plt.xlim((0, size-1))
        plt.ylim((0, size-1))
        plt.grid()
    elif mode == 'channels':
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, title in zip(range(3), ['Food','Head','Body']):
            axes[i].set_title(title)
            axes[i].imshow(envs[env_idx, i, :, :].cpu().numpy())
            axes[i].grid()
            axes[i].set_xlim((0, size-1))
            axes[i].set_ylim((0, size-1))
    elif mode == 'multi':
        n = len(env_idx)
        fig, axes = plt.subplots(1, n, figsize=(n*5, 5))

        for i, env_i in enumerate(env_idx):
            img = (envs[env_i, BODY_CHANNEL, :, :].cpu().numpy() > 0) * 0.5
            img += envs[env_i, HEAD_CHANNEL, :, :].cpu().numpy() * 0.5
            img += envs[env_i, FOOD_CHANNEL, :, :].cpu().numpy() * 1.5
            axes[i].imshow(img, vmin=0, vmax=1.5)
            axes[i].set_xlim((0, size - 1))
            axes[i].set_ylim((0, size - 1))
            axes[i].grid()
    else:
        raise Exception


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


class TestSingleSnakeEnv(unittest.TestCase):
    def test_multiple_envs(self):
        num_envs = 100
        num_steps = 100
        env = SingleSnake(num_envs=num_envs, size=size)
        actions = torch.randint(4, size=(num_steps, num_envs)).long().to(DEFAULT_DEVICE)

        t0 = time()
        for i, a in enumerate(actions):
            if visualise:
                plot_envs(env.envs)
                plt.show()

            observations, reward, done, info = env.step(a)
            env.reset(done)
            env_consistency(env.envs)

        t = time() - t0
        print(f'Ran {num_envs*num_steps} actions in {t}s = {num_envs*num_steps/t} actions/s')

    def test_setup(self):
        n = 97
        env = SingleSnake(num_envs=n, size=size)
        env_consistency(env.envs)
        expected_body_sum = env.initial_snake_length * (env.initial_snake_length + 1) / 2
        self.assertTrue(torch.all(body(env.envs).view(n, -1).sum(dim=-1) == expected_body_sum))

    def test_reset(self):
        env = SingleSnake(num_envs=1, size=size)
        env_consistency(env.envs)
        env.reset(torch.Tensor([1]).to(DEFAULT_DEVICE))
        env_consistency(env.envs)

    def test_loop_movement(self):
        pass

    def test_basic_movement(self):
        env = SingleSnake(num_envs=1, size=size, manual_setup=True)
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

            if torch.any(done):
                # The given actions shouldn't cause a death
                assert False

        if visualise:
            plot_envs(env.envs)
            plt.show()

    def test_eat_food(self):
        env = SingleSnake(num_envs=1, size=size, manual_setup=True)
        env.envs = get_test_env(size, 'up').to(DEFAULT_DEVICE)
        actions = torch.Tensor([0, 3, 3, 0, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE)

        initial_size = body(env.envs).max()

        for i, a in enumerate(actions):
            if visualise:
                plot_envs(env.envs)
                plt.show()

            observations, reward, done, info = env.step(a)

            if torch.any(done):
                # The given actions shouldn't cause a death
                assert False

        final_size = body(env.envs).max()
        self.assertGreater(final_size, initial_size)

        # Food is created again after being eaten
        num_food = food(env.envs).sum()
        print(num_food, 1)
        self.assertEqual(num_food, 1)

        # Check overall consistency
        env_consistency(env.envs)

        if visualise:
            plot_envs(env.envs)
            plt.show()

    def test_hit_boundary(self):
        env = SingleSnake(num_envs=1, size=size, manual_setup=True)
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
        env = SingleSnake(num_envs=1, size=size, manual_setup=True)
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
        env = SingleSnake(num_envs=1, size=size, manual_setup=True)
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

            if torch.any(done):
                # The given actions shouldn't cause a death
                assert False

        if visualise:
            plot_envs(env.envs)
            plt.show()
