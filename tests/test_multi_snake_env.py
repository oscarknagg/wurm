import unittest
import torch

from wurm.envs import MultiSnakeEnvironments
from wurm.utils import head
from config import DEFAULT_DEVICE


print_envs = False
size = 12


def get_test_env():
    env = MultiSnakeEnvironments(num_envs=1, num_snakes=2, size=size, manual_setup=True)

    # Food
    env.envs[0, 0, 1, 1] = 1
    # Snake 1
    env.envs[0, 1, 5, 5] = 1
    env.envs[0, 2, 5, 5] = 3
    env.envs[0, 2, 4, 5] = 2
    env.envs[0, 2, 4, 4] = 1
    # Snake 2
    env.envs[0, 3, 8, 7] = 1
    env.envs[0, 4, 8, 7] = 3
    env.envs[0, 4, 8, 8] = 2
    env.envs[0, 4, 8, 9] = 1

    return env


class TestMultiSnakeEnv(unittest.TestCase):
    def test_basic_movement(self):
        env = get_test_env()
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
                [8, 6],
                [8, 7]
            ]),
        ]

        print()
        if print_envs:
            print(env._bodies)

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
                print(i_agent, head_position)

            if print_envs:
                print('='*10)
                print(env._bodies)
                print('DONES:')
                print(dones)

            print()
            if any(done for agent, done in dones.items()):
                # These actions shouldn't cause any deaths
                assert False

    def test_self_collision(self):
        pass

    def test_other_snake_collision(self):
        pass

    def test_eat_food(self):
        pass

    def test_post_death(self):
        # Test nothing breaks when 1 snake dies and the other continues
        pass

    def test_reset(self):
        # Environment resets when all snakes are dead
        pass
