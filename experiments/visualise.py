import numpy as np
from itertools import count
from collections import namedtuple
import argparse
from time import time, sleep
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from wurm.envs import SingleSnakeEnvironments, SimpleGridworld
from wurm.agents import A2C as Snake2C
from wurm.utils import env_consistency, CSVLogger
from config import BODY_CHANNEL, HEAD_CHANNEL, FOOD_CHANNEL, PATH, EPS


SEED = 543
RENDER = False
LOG_INTERVAL = 100
MAX_GRAD_NORM = 0.5
FPS = 24

parser = argparse.ArgumentParser()
parser.add_argument('--env')
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--observation', default='default', type=str)
parser.add_argument('--coord-conv', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--reward-shaping', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--render', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--size', default=9, type=int)
parser.add_argument('--update-steps', default=20, type=int)
parser.add_argument('--num-envs', default=1, type=int)
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--entropy', default=0.0, type=float)
args = parser.parse_args()
argstring = '__'.join([f'{k}={v}' for k, v in args.__dict__.items()])
print(argstring)


Transition = namedtuple('Transition', ['action', 'log_prob', 'value', 'reward', 'done', 'entropy'])


class A2C(nn.Module):
    def __init__(self, num_actions: int, num_inputs: int = 4):
        super(A2C, self).__init__()
        self.num_actions = num_actions
        self.affine1 = nn.Linear(num_inputs, 128)
        self.action_head = nn.Linear(128, self.num_actions)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


if args.env == 'gridworld':
    size = args.size
    env = SimpleGridworld(num_envs=1, size=size, start_location=(size//2, size//2), observation_mode=args.observation, device=args.device)
    if args.observation == 'positions':
        model = A2C(4).to(args.device)
    else:
        model = Snake2C(in_channels=2, size=size, coord_conv=args.coord_conv).to(args.device)
elif args.env == 'snake':
    size = args.size
    env = SingleSnakeEnvironments(num_envs=1, size=size, device=args.device, observation_mode=args.observation)
    if args.observation == 'positions':
        model = A2C(4).to(args.device)
    if args.observation.startswith('partial_'):
        observation_size = int(args.observation.split('_')[-1])
        observation_width = 2 * observation_size + 1
        model = A2C(4, num_inputs=3 * (observation_width ** 2)).to(args.device)
    else:
        model = Snake2C(
            in_channels=1 if args.observation == 'one_channel' else 3, size=size, coord_conv=args.coord_conv).to(args.device)
else:
    raise ValueError('Unrecognised environment')


model.load_state_dict(torch.load(f'{PATH}/models/{argstring}.pt'))
model.eval()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()

running_length = None
running_self_collisions = None
running_edge_collisions = None
running_reward_rate = None
running_entropy = None

saved_transitions = []

episode_length = 0
num_episodes = 0
num_steps = 0
logger = CSVLogger(filename=f'{PATH}/logs/{argstring}.csv')

t0 = time()
state = env.reset()
for i_step in count(1):
    env.render()
    sleep(1./FPS)

    probs, state_value = model(state)
    m = Categorical(probs)
    entropy = m.entropy().mean()
    action = m.sample().clone().long()

    state, reward, done, info = env.step(action)
    if args.env == 'snake':
        env_consistency(env.envs[~done.squeeze(-1)])

    if args.reward_shaping:
        # Hacky reward shaping
        size = torch.Tensor([env.size, ] * env.num_envs).long().to(env.device)
        head_idx = env.envs[:, HEAD_CHANNEL, :, :].view(env.num_envs, env.size ** 2).argmax(dim=-1)
        food_idx = env.envs[:, FOOD_CHANNEL, :, :].view(env.num_envs, env.size ** 2).argmax(dim=-1)
        head_pos = torch.stack((head_idx // size, head_idx % size)).float().t()
        food_pos = torch.stack((food_idx // size, food_idx % size)).float().t()
        food_closeness_reward = torch.clamp(0.1 - torch.norm(head_pos - food_pos, p=1, dim=-1) * 0.01, 0, 0.1)

    saved_transitions.append(Transition(action, m.log_prob(action), state_value, reward, done, entropy))

    env.reset(done)


