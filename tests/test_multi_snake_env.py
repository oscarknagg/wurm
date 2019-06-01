import unittest
import pytest
import torch
from time import sleep, time
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from wurm.envs import MultiSnake
from wurm.utils import head, body, food, determine_orientations
from config import DEFAULT_DEVICE


print_envs = False
render_envs = False
render_sleep = 1
size = 12
torch.random.manual_seed(0)


def get_test_env(num_envs=1):
    env = MultiSnake(num_envs=num_envs, num_snakes=2, size=size, manual_setup=True)

    # Snake 1
    env.heads[0, 0, 5, 5] = 1
    env.bodies[0, 0, 5, 5] = 4
    env.bodies[0, 0, 4, 5] = 3
    env.bodies[0, 0, 4, 4] = 2
    env.bodies[0, 0, 4, 3] = 1
    # Snake 2
    env.heads[1, 0, 8, 7] = 1
    env.bodies[1, 0, 8, 7] = 4
    env.bodies[1, 0, 8, 8] = 3
    env.bodies[1, 0, 8, 9] = 2
    env.bodies[1, 0, 9, 9] = 1

    _envs = torch.cat([
        env.foods.repeat_interleave(env.num_snakes, dim=0),
        env.heads,
        env.bodies
    ], dim=1)

    env.orientations = determine_orientations(_envs)

    return env


def print_or_render(env):
    if print_envs:
        print('='*30)
        print(env.foods)
        print('-'*30)
        print(env.heads)
        print('-' * 30)
        print(env.bodies)
        print('-' * 30)

    if render_envs:
        env.render()
        sleep(render_sleep)


class TestMultiSnakeEnv(unittest.TestCase):
    def test_random_actions(self):
        num_envs = 100
        num_steps = 100
        # Create some environments and run random actions for N steps, checking for consistency at each step
        env = MultiSnake(num_envs=num_envs, num_snakes=2, size=size, manual_setup=False, verbose=True,
                         render_args={'num_rows': 5, 'num_cols': 5, 'size': 128},
                         )
        env.check_consistency()

        all_actions = {
            'agent_0': torch.randint(4, size=(num_steps, num_envs)).long().to(DEFAULT_DEVICE),
            'agent_1': torch.randint(4, size=(num_steps, num_envs)).long().to(DEFAULT_DEVICE),
        }

        t0 = time()
        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }
            observations, reward, done, info = env.step(actions)

            env.reset(done['__all__'])
            env.check_consistency()
            print()

        t = time() - t0
        print(f'Ran {num_envs * num_steps} actions in {t}s = {num_envs * num_steps / t} actions/s')

    def test_random_actions_with_boost(self):
        # num_envs = 1024*8
        num_envs = 256
        num_steps = 200
        num_snakes = 4
        # Create some environments and run random actions for N steps, checking for consistency at each step
        env = MultiSnake(num_envs=num_envs, num_snakes=num_snakes, size=25, manual_setup=False, boost=True, verbose=True,
                         render_args={'num_rows': 1, 'num_cols': 2, 'size': 256},
                         respawn_mode='any', food_mode='random_rate', boost_cost_prob=0.25,
                         observation_mode='partial_5', food_on_death_prob=0.33, food_rate=2.5e-4
                         )
        env.check_consistency()

        all_actions = {
            f'agent_{i}': torch.randint(8, size=(num_steps, num_envs)).long().to(DEFAULT_DEVICE) for i in
            range(num_snakes)
        }

        t0 = time()
        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }
            observations, reward, done, info = env.step(actions)

            env.reset(done['__all__'])
            env.check_consistency()
            print()

        t = time() - t0
        print(f'Ran {num_envs * num_steps} actions in {t}s = {num_envs * num_steps / t} actions/s')

    def test_basic_movement(self):
        env = get_test_env()

        # Add food
        env.foods[0, 0, 1, 1] = 1

        all_actions = {
            'agent_0': torch.tensor([1, 2, 1, 1, 0, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 3, 2, 1, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_head_positions = [
            torch.tensor([
                [5, 4],
                [4, 4],
                [4, 3],
                [4, 2],
                [5, 2],
                [5, 3]
            ]),
            torch.tensor([
                [9, 7],
                [9, 6],
                [9, 5],
                [8, 5],
                [8, 4],
                [9, 4]
            ]),
        ]

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            env.check_consistency()

            for i_agent in range(env.num_snakes):
                head_position = torch.tensor([
                    env.heads[i_agent, 0].flatten().argmax() // size, env.heads[i_agent, 0].flatten().argmax() % size
                ])
                print(i, head_position, expected_head_positions[i_agent][i])
                self.assertTrue(torch.equal(head_position, expected_head_positions[i_agent][i]))

            print_or_render(env)

            if any(done for agent, done in dones.items()):
                # These actions shouldn't cause any deaths
                assert False

    def test_edge_collision(self):
        env = get_test_env()
        env.food_on_death_prob = 1

        # Add food so agent_0 eats it and grows
        env.foods[0, 0, 1, 1] = 1

        all_actions = {
            'agent_0': torch.tensor([1, 1, 1, 1, 1, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 2, 6, 2, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            env.check_consistency()
            print(i, dones)

            print_or_render(env)
            print('='*30)

            if i >= 4:
                self.assertEqual(dones['agent_0'].item(), 1)
            else:
                self.assertEqual(dones['agent_0'].item(), 0)

            if i >= 2:
                self.assertEqual(dones['agent_1'].item(), 1)
            else:
                self.assertEqual(dones['agent_1'].item(), 0)

    def test_self_collision(self):
        env = get_test_env()
        env.food_on_death_prob = 1

        # Add food so agent_0 eats it and grows
        env.foods[0, 0, 4, 3] = 1

        all_actions = {
            'agent_0': torch.tensor([1, 2, 1, 1, 0, 3, 2, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 3, 2, 1, 0, 0, 1]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            env.check_consistency()

            print_or_render(env)

            if i >= 6:
                self.assertEqual(dones['agent_0'].item(), 1)
            else:
                self.assertEqual(dones['agent_0'].item(), 0)

        # # Check some food has been created on death
        # self.assertGreaterEqual(env.foods[:, 0].sum().item(), 2)

    def test_other_snake_collision(self):
        # Actions and snakes are arranged so agent_1 collides with agent_0 and dies
        env = get_test_env()
        env.foods[0, 0, 1, 1] = 1
        env.food_on_death_prob = 1

        all_actions = {
            'agent_0': torch.tensor([1, 2, 3, 3, 3, 3, 3, 2]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([1, 2, 2, 2, 2, 2, 2, 2]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            env.check_consistency()

            if i >= 4:
                self.assertEqual(dones['agent_1'].item(), 1)
            else:
                self.assertEqual(dones['agent_1'].item(), 0)

            print_or_render(env)

        # Check some food has been created on death
        self.assertGreaterEqual(env.foods[:, 0].sum().item(), 2)

    def test_eat_food(self):
        env = get_test_env()

        # Add food
        env.foods[0, 0, 9, 7] = 1

        all_actions = {
            'agent_0': torch.tensor([1, 2, 1, 1, 0, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 3, 2, 1, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(6):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            env.check_consistency()

            print_or_render(env)

            print(i, rewards)

            # Check reward given when expected
            if i == 0:
                self.assertEqual(rewards['agent_1'].item(), 1)
            else:
                self.assertEqual(rewards['agent_1'].item(), 0)

            if any(done for agent, done in dones.items()):
                # These actions shouldn't cause any deaths
                assert False

        # Check snake sizes. Expect agent_1: 4, agent_2: 5
        snake_sizes = env.bodies.view(1, 2, -1).max(dim=2)[0].long()
        self.assertTrue(torch.equal(snake_sizes, torch.tensor([[4, 5]]).to(DEFAULT_DEVICE)))

        # Check food has been removed
        self.assertEqual(env.foods[0, 0, 9, 7].item(), 0)

        # Check new food has been created
        self.assertEqual(env.foods.sum().item(), 1)

    def test_create_envs(self):
        # Create a large number of environments and check consistency
        env = MultiSnake(num_envs=512, num_snakes=2, size=size, manual_setup=False)
        env.check_consistency()

        _envs = torch.cat([
            env.foods.repeat_interleave(env.num_snakes, dim=0),
            env.heads,
            env.bodies
        ], dim=1)

        orientations = determine_orientations(_envs)
        self.assertTrue(torch.equal(env.orientations, orientations))

    def test_reset(self):
        num_snakes = 2

        # agent_1 dies by other-collision, agent_0 dies by edge-collision
        env = get_test_env(num_envs=1)
        env.foods[:, 0, 1, 1] = 1

        all_actions = {
            'agent_0': torch.tensor([1, 2, 3, 3, 3, 3, 3, 3, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 2, 2, 2, 2, 2, 2, 2]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            env.reset(dones['__all__'])
            print(i, dones)

            env.check_consistency()

            print_or_render(env)

        # Both snakes should've died and hence the environment should've reset
        self.assertTrue(torch.all(env.bodies.view(1, num_snakes, -1).max(dim=-1)[0] == env.initial_snake_length))

    def test_agent_observations(self):
        # Test that own snake appears green, others appear blue
        env = get_test_env(num_envs=1)
        env.foods[:, 0, 1, 1] = 1

        obs_0 = env._observe_agent(0)
        obs_1 = env._observe_agent(1)

        # Check that own agent appears green and other appears blue
        self.assertTrue(torch.allclose(obs_0[0, :, 4, 5]*255, env.self_colour.float()/2))
        self.assertTrue(torch.allclose(obs_0[0, :, 8, 8]*255, env.other_colour.float()/2))

        self.assertTrue(torch.allclose(obs_1[0, :, 4, 5]*255, env.other_colour.float()/2))
        self.assertTrue(torch.allclose(obs_1[0, :, 8, 8]*255, env.self_colour.float()/2))

    def test_boost_through_food(self):
        # Test boosting through a food
        env = get_test_env(num_envs=1)
        env.boost = True
        env.foods[:, 0, 6, 5] = 1
        env.boost_cost_prob = 0

        all_actions = {
            'agent_0': torch.tensor([4, 1, 2]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            env.reset(dones['__all__'])

            env.check_consistency()

            print_or_render(env)

            if i == 0:
                self.assertEqual(rewards['agent_0'].item(), 1)

    def test_boost_leaves_food(self):
        # Test boosting through a food
        env = get_test_env(num_envs=1)
        env.boost = True
        env.boost_cost_prob = 1
        env.foods[:, 0, 1, 5] = 0

        all_actions = {
            'agent_0': torch.tensor([4, 1, 2]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            env.reset(dones['__all__'])

            env.check_consistency()

            print_or_render(env)

            if i == 0:
                self.assertEqual(rewards['agent_0'].item(), -1)

        self.assertEqual(env.foods[0, 0, 4, 4].item(), 1)

    def test_cant_boost_until_size_4(self):
        # Create a size 3 snake and try boosting with it
        env = MultiSnake(num_envs=1, num_snakes=2, size=size, manual_setup=True, boost=True)
        env.foods[:, 0, 1, 1] = 1
        # Snake 1
        env.heads[:, 0, 5, 5] = 1
        env.bodies[:, 0, 5, 5] = 3
        env.bodies[:, 0, 4, 5] = 2
        env.bodies[:, 0, 4, 4] = 1
        # Snake 2
        env.heads[:, 0, 8, 7] = 1
        env.bodies[:, 0, 8, 7] = 3
        env.bodies[:, 0, 8, 8] = 2
        env.bodies[:, 0, 8, 9] = 1

        # Get orientations manually
        _envs = torch.cat([
            env.foods.repeat_interleave(env.num_snakes, dim=0),
            env.heads,
            env.bodies
        ], dim=1)

        env.orientations = determine_orientations(_envs)

        expected_head_positions = torch.tensor([
            [6, 5],
            [6, 4],
            [5, 4],
        ])

        all_actions = {
            'agent_0': torch.tensor([4, 1, 2]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            env.reset(dones['__all__'])
            print(dones)

            env.check_consistency()

            for i_agent in range(env.num_snakes):
                _env = torch.cat([
                    env.foods,
                    env.heads[i_agent].unsqueeze(0),
                    env.bodies[i_agent].unsqueeze(0)
                ], dim=1)

                head_position = torch.tensor([
                    head(_env)[0, 0].flatten().argmax() // size, head(_env)[0, 0].flatten().argmax() % size
                ])

                # if i_agent == 0:
                #     self.assertTrue(torch.equal(expected_head_positions[i], head_position))

            print_or_render(env)

    def test_boost_cost(self):
        env = get_test_env(num_envs=1)
        env.boost = True
        env.boost_cost_prob = 1
        env.foods[:, 0, 1, 1] = 1

        all_actions = {
            'agent_0': torch.tensor([4, 1, 2]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            env.reset(dones['__all__'])

            env.check_consistency()

            print_or_render(env)

            # Check snake sizes. Expect agent_1: 3, agent_2: 4
            snake_sizes = env.bodies.view(1, 2, -1).max(dim=2)[0].long()
            self.assertTrue(torch.equal(snake_sizes, torch.tensor([[3, 4]]).to(DEFAULT_DEVICE)))

            if i == 0:
                self.assertEqual(rewards['agent_0'].item(), -1)

    def test_many_snakes(self):
        num_envs = 50
        num_steps = 10
        num_snakes = 4
        env = MultiSnake(num_envs=num_envs, num_snakes=num_snakes, size=size, manual_setup=False, boost=True)
        env.check_consistency()

        all_actions = {
            f'agent_{i}': torch.randint(8, size=(num_steps, num_envs)).long().to(DEFAULT_DEVICE) for i in range(num_snakes)
        }

        for i in range(all_actions['agent_0'].shape[0]):
            # env.render()
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }
            observations, reward, done, info = env.step(actions)
            env.reset(done['__all__'])
            env.check_consistency()

    def test_boost_rendering(self):
        # Test boosting through a food
        env = get_test_env(num_envs=1)
        env.boost = True
        env.foods[:, 0, 1, 5] = 1
        print('=====')
        print(env.self_colour/2)
        print(env.self_colour)
        all_actions = {
            'agent_0': torch.tensor([4, 1, 2]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            env.reset(dones['__all__'])

            env.check_consistency()

            print_or_render(env)
            print(env.boost_this_step)

    def test_respawn_mode_any(self):
        # Create an env where there is no room to respawn and see if we get an exception
        env = get_test_env()
        env.respawn_mode = 'any'

        # Add food to block respawn
        for i in range(2, 9, 2):
            for j in range(2, 9, 2):
                env.foods[0, 0, i, j] = 1

        all_actions = {
            'agent_0': torch.tensor([1, 1, 1, 1, 2, 2, 2, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 1, 0, 0, 0, 0, 0, 1]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        print_or_render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            pprint(dones)
            env.reset(dones['__all__'])
            env.check_consistency()

    def test_partial_observations(self):
        num_envs = 2
        num_snakes = 4
        observation_mode = 'partial_3'
        env = MultiSnake(num_envs=num_envs, num_snakes=num_snakes, size=size, manual_setup=False, boost=True,
                         observation_mode=observation_mode,
                         render_args={'num_rows': 1, 'num_cols': 2, 'size': 256},
                         )
        env.check_consistency()

        observations = env._observe(observation_mode)
        if render_envs:
            fig, axes = plt.subplots(2, 2)
            i = 0
            # Show all the observations of the agent in the first env
            for k, v in observations.items():
                axes[i // 2, i % 2].imshow(v[0].permute(1, 2, 0).cpu().numpy())
                i += 1

            plt.show()

            env.render()
            sleep(5)
