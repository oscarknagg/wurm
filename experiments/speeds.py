import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time
import argparse
from wurm.envs import MultiSnake
from config import DEFAULT_DEVICE


parser = argparse.ArgumentParser()
parser.add_argument('--num-agents', type=int, default=10)
parser.add_argument('--size', type=int, default=36)
args = parser.parse_args()

num_envs = np.logspace(4, 12, 9, base=2).astype(int)
num_steps = 10

fps = []

for n in num_envs:
    env = MultiSnake(num_envs=n, num_snakes=args.num_agents, size=args.size, manual_setup=False, boost=True,
                     verbose=False, device='cuda', respawn_mode='any')

    all_actions = {
        f'agent_{i}': torch.randint(8, size=(num_steps, n)).long().to(DEFAULT_DEVICE) for i in range(args.num_agents)
    }

    env_steps = 0
    t0 = time()
    for i in range(num_steps):
        # print(i)
        actions = {
            agent: agent_actions[i] for agent, agent_actions in all_actions.items()
        }
        observations, reward, done, info = env.step(actions)
        env.reset(done['__all__'])
        env.check_consistency()
        env_steps += n

    t = time()

    print(n, env_steps/(t - t0))

    fps.append((n, env_steps/(t - t0)))
