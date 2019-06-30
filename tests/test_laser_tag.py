import unittest
import torch
from time import sleep
import matplotlib.pyplot as plt

from wurm.envs import LaserTag
from wurm.envs.laser_tag.maps import Small2, Small3, Small4
from wurm import observations
from tests._laser_trajectories import expected_laser_trajectories
from config import DEFAULT_DEVICE


render_envs = False
size = 9
render_sleep = 0.5


def get_test_env(num_envs=2):
    # Same as small2 map from the Deepmind paper
    env = LaserTag(num_envs, 2, height=size, width=size, map_generator=Small2(DEFAULT_DEVICE), manual_setup=True)
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
    env.pathing[:, :, :1, :] = 1
    env.pathing[:, :, -1:, :] = 1
    env.pathing[:, :, :, :1] = 1
    env.pathing[:, :, :, -1:] = 1

    env.respawns = torch.zeros((num_envs, 1, size, size), dtype=torch.uint8, device=DEFAULT_DEVICE, requires_grad=False)
    env.respawns[:, :, 1, 1] = 1
    env.respawns[:, :, 1, 7] = 1
    env.respawns[:, :, 7, 1] = 1
    env.respawns[:, :, 7, 7] = 1

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

            obs, rewards, dones, info = env.step(actions)
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
        raise Exception

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
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 3, 3, 3, 3, 3, 3, 1, 3, 0, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            if i == 10:
                # This checks that the space behind agent_1 doesn't contain a laser
                self.assertEqual(env.lasers[0, 0, 7, 1].item(), 0)

            render(env)

    def test_death_and_respawn(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 3, 3, 3, 3, 3, 3, 0, 0, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            env.reset(dones['__all__'])

            render(env)

    def test_asymettric_map(self):
        # env = LaserTag(num_envs=1, num_agents=2, height=9, width=16, map_generator=Small3(DEFAULT_DEVICE),
        #                device=DEFAULT_DEVICE)
        env = LaserTag(num_envs=1, num_agents=2, height=14, width=22, map_generator=Small4(DEFAULT_DEVICE),
                       device=DEFAULT_DEVICE)
        if render_envs:
            env.render()
            sleep(5)

    def test_observations(self):
        obs_fn = observations.RenderObservations()
        env = LaserTag(num_envs=1, num_agents=2, height=9, width=9,
                       map_generator=Small2(DEFAULT_DEVICE), observation_fn=obs_fn,
                       device=DEFAULT_DEVICE)

        agent_obs = obs_fn.observe(env)
        for agent, obs in agent_obs.items():
            if render_envs:
                obs_npy = obs.permute((2, 3, 1, 0))[:, :, :, 0].cpu().numpy()
                plt.imshow(obs_npy)
                plt.show()

    def test_partial_observations(self):
        obs_fn = observations.FirstPersonCrop(
            first_person_rotation=True,
            in_front=7,
            behind=2,
            side=3
        )
        # obs_fn = observations.FirstPersonCrop(height=5, width=5)

        # env = LaserTag(num_envs=1, num_agents=2, height=9, width=9,
        #                map_generator=Small2(DEFAULT_DEVICE), observation_fn=obs_fn,
        #                device=DEFAULT_DEVICE)
        env = get_test_env(num_envs=1)

        render_envs = True
        agent_obs = obs_fn.observe(env)
        for agent, obs in agent_obs.items():
            if render_envs:
                obs_npy = obs.permute((2, 3, 1, 0))[:, :, :, 0].cpu().numpy()
                plt.imshow(obs_npy)
                plt.show()

        env.render()
        sleep(5)

    def test_create_envs(self):
        pass

    def test_render(self):
        env = get_test_env()

        if render_envs:
            env.render()
            sleep(5)
