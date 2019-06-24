import unittest
import pytest
import torch
from time import sleep

from wurm.envs import LaserTag
from config import DEFAULT_DEVICE

render_envs = True
size = 9
render_sleep = 1


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


def render(env):
    if render_envs:
        env.render()
        sleep(render_sleep)


class TestLaserTag(unittest.TestCase):
    def _test_action_sequence(self, env, all_actions, expected_orientations, expected_x, expected_y):
        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            render(env)
            self.assertTrue(torch.equal(env.x.cpu(), expected_x[i]))
            self.assertTrue(torch.equal(env.y.cpu(), expected_y[i]))
            if expected_orientations is not None:
                self.assertTrue(torch.equal(env.orientations.cpu(), expected_orientations[i]))

    def test_basic_movement(self):
        """2 agents rotate completely on the spot then move in a circle."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([1, 1, 1, 1, 3, 2, 3, 2, 3, 2, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        expected_orientations = torch.tensor([
            [1, 2, 3, 0, 0, 3, 3, 2, 2, 1, 1],
            [1, 0, 3, 2, 2, 1, 1, 0, 0, 3, 3],
        ]).t()
        expected_x = torch.tensor([
            [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
            [7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7],
        ]).t()
        expected_y = torch.tensor([
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1],
            [7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7],
        ]).t()

        self._test_action_sequence(env, all_actions, expected_orientations, expected_x, expected_y)

    def test_move_backward(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([3, 4]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([3, 4]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_x = torch.tensor([
            [2, 1],
            [6, 7],
        ]).t()
        expected_y = torch.tensor([
            [1, 1],
            [7, 7],
        ]).t()
        expected_orientations = torch.tensor([
            [0, 0],
            [2, 2],
        ]).t()

        self._test_action_sequence(env, all_actions, expected_orientations, expected_x, expected_y)

    def test_strafe(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([6, 5]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([6, 5]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_x = torch.tensor([
            [1, 1],
            [7, 7],
        ]).t()
        expected_y = torch.tensor([
            [2, 1],
            [6, 7],
        ]).t()
        expected_orientations = torch.tensor([
            [0, 0],
            [2, 2],
        ]).t()

        self._test_action_sequence(env, all_actions, expected_orientations, expected_x, expected_y)

    def test_pathing(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 5, 6, 6, 3, 3, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([4, 3, 0, 3, 2, 3, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_x = torch.tensor([
            [1, 1, 1, 1, 2, 2, 2],
            [7, 6, 6, 5, 5, 5, 5],
        ]).t()
        expected_y = torch.tensor([
            [1, 1, 2, 3, 3, 3, 3],
            [7, 7, 7, 7, 7, 6, 6]
        ]).t()

        self._test_action_sequence(env, all_actions, None, expected_x, expected_y)

    def test_firing(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([7, 6, 7, 6, 7, 2, 7]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 7, 2, 7, 0, 0, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        render(env)
        print('-'*50)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            render(env)
            print('-' * 50)

    def test_render(self):
        env = get_test_env()

        if render_envs:
            env.render()
            sleep(5)
