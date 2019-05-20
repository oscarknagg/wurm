import unittest
import pytest
import torch
from time import sleep

from wurm.envs import MultiSnake
from wurm.utils import head, body, food
from config import DEFAULT_DEVICE


print_envs = False
render_envs = True
render_sleep = 0.5
size = 12


def get_test_env():
    env = MultiSnake(num_envs=1, num_snakes=2, size=size, manual_setup=True)

    # Snake 1
    env.envs[0, 1, 5, 5] = 1
    env.envs[0, 2, 5, 5] = 4
    env.envs[0, 2, 4, 5] = 3
    env.envs[0, 2, 4, 4] = 2
    env.envs[0, 2, 4, 3] = 1
    # Snake 2
    env.envs[0, 3, 8, 7] = 1
    env.envs[0, 4, 8, 7] = 4
    env.envs[0, 4, 8, 8] = 3
    env.envs[0, 4, 8, 9] = 2
    env.envs[0, 4, 9, 9] = 1

    return env


class TestMultiSnakeEnv(unittest.TestCase):
    @pytest.mark.skip()
    def test_basic_movement(self):
        env = get_test_env()

        # Add food
        env.envs[0, 0, 1, 1] = 1

        all_actions = {
            'agent_0': torch.Tensor([1, 2, 1, 1, 0, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.Tensor([0, 1, 3, 2, 1, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_head_positions = [
            torch.Tensor([
                [5, 4],
                [4, 4],
                [4, 3],
                [4, 2],
                [5, 2],
                [5, 3]
            ]),
            torch.Tensor([
                [9, 7],
                [9, 6],
                [9, 5],
                [8, 5],
                [8, 4],
                [9, 4]
            ]),
        ]

        print()
        if print_envs:
            print(env._bodies)

        if render_envs:
            env.render()
            sleep(render_sleep)

        for i in range(6):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            env.check_consistency()

            for i_agent in range(env.num_snakes):
                head_channel = env.head_channels[i_agent]
                body_channel = env.body_channels[i_agent]
                _env = env.envs[:, [0, head_channel, body_channel], :, :]

                head_position = torch.Tensor([
                    head(_env)[0, 0].flatten().argmax() // size, head(_env)[0, 0].flatten().argmax() % size
                ])
                self.assertTrue(torch.equal(head_position, expected_head_positions[i_agent][i]))
                # print(i_agent, head_position)

            if print_envs:
                print('='*10)
                print(env._bodies)
                print('DONES:')
                print(dones)
                print()

            if render_envs:
                env.render()
                sleep(render_sleep)

            if any(done for agent, done in dones.items()):
                # These actions shouldn't cause any deaths
                assert False

    def test_self_collision(self):
        pass

    def test_other_snake_collision(self):
        pass

    def test_eat_food(self):
        env = get_test_env()

        # Add food
        env.envs[0, 0, 9, 7] = 1

        all_actions = {
            'agent_0': torch.Tensor([1, 2, 1, 1, 0, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.Tensor([0, 1, 3, 2, 1, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print()
        if print_envs:
            print(env._bodies)

        if render_envs:
            env.render()
            sleep(render_sleep)

        for i in range(6):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            env.check_consistency()

            if print_envs:
                print('=' * 10)
                print(env._bodies)
                print('DONES:')
                print(dones)
                print()

            # Check reward given when expected
            if i == 0:
                self.assertEqual(rewards['agent_1'].item(), 1)

            if render_envs:
                env.render()
                sleep(render_sleep)

            if any(done for agent, done in dones.items()):
                # These actions shouldn't cause any deaths
                assert False

        # Check snake sizes. Expect agent_1: 4, agent_2: 5
        snake_sizes = env._bodies.view(1, 2, -1).max(dim=2)[0]
        self.assertTrue(torch.equal(snake_sizes, torch.Tensor([[4, 5]]).to(DEFAULT_DEVICE)))

        # Check food has been removed
        self.assertEqual(env.envs[0, 0, 9, 7].item(), 0)

        # Check new food has been created
        self.assertEqual(food(env.envs).sum().item(), 1)

    def test_post_death(self):
        # Test nothing breaks when 1 snake dies and the other continues
        pass

    def test_food_creation_on_death(self):
        pass

    def test_reset(self):
        # Environment resets when all snakes are dead
        pass

    def test_agent_observations(self):
        # Test that own snake appears green, others appear blue
        pass
