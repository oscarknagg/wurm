import unittest
import pytest
import torch
from time import sleep

from wurm.envs import LaserTag
from tests._laser_trajectories import expected_laser_trajectories
from config import DEFAULT_DEVICE


render_envs = False
size = 9
render_sleep = 0.75


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
    def _test_action_sequence(self, env, all_actions, expected_orientations=None, expected_x=None, expected_y=None,
                              expected_hp=None, expected_reward=None, expected_done=None):
        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            render(env)

            if expected_x is not None:
                self.assertTrue(torch.equal(env.x.cpu(), expected_x[i]))
            if expected_y is not None:
                self.assertTrue(torch.equal(env.y.cpu(), expected_y[i]))
            if expected_orientations is not None:
                self.assertTrue(torch.equal(env.orientations.cpu(), expected_orientations[i]))
            if expected_hp is not None:
                self.assertTrue(torch.equal(env.hp.cpu(), expected_hp[i]))
            if expected_reward is not None:
                self.assertTrue(torch.equal(env.rewards.cpu(), expected_reward[i]))
            if expected_done is not None:
                self.assertTrue(torch.equal(env.dones.cpu(), expected_done[i]))

    def test_random_actions(self):
        """Tests a very large number of random actions and checks for environment consistency
        instead of any particular expected trajectory."""
        pass

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

    def test_wall_pathing(self):
        """Test agents can't walk through walls."""
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

    def test_agent_agent_pathing(self):
        """Test agents can't walk through each other."""
        pass

    def test_firing(self):
        """Tests that laser trajectories are calculated correctly."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([7, 6, 7, 6, 7, 2, 7]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 7, 2, 7, 0, 0, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            # Laser trajectories were verified manually then saved to another file because they are very verbose
            self.assertTrue(torch.equal(env.lasers, expected_laser_trajectories[i]))
            render(env)

    def test_being_hit(self):
        """Tests that agent-laser collision is calculated correctly, hp is deducted
        and reward is given."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 7, 7, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 3, 3, 3, 3, 3, 3, 0, 0, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_hp = torch.tensor([
            [2, ] * all_actions['agent_0'].shape[0],
            [2, ] * (all_actions['agent_0'].shape[0] - 2) + [1, 0],
        ]).t()
        expected_reward = torch.tensor([
            [0, ] * (all_actions['agent_0'].shape[0] - 2) + [1, 1],
            [0, ] * (all_actions['agent_0'].shape[0] - 2) + [0, 0],
        ]).t().float()
        expected_done = torch.tensor([
            [0, ] * all_actions['agent_0'].shape[0],
            [0, ] * (all_actions['agent_0'].shape[0] - 1) + [1],
        ]).t().byte()

        self._test_action_sequence(env, all_actions, expected_hp=expected_hp, expected_reward=expected_reward,
                                   expected_done=expected_done)

    def test_cant_shoot_through_agents(self):
        pass

    def test_death_and_respawn(self):
        pass

    def test_partial_observations(self):
        pass

    def test_render(self):
        env = get_test_env()

        if render_envs:
            env.render()
            sleep(5)