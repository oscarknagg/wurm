import gym
import numpy as np
from itertools import count
from collections import namedtuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from wurm.envs import SingleSnakeEnvironments, SimpleGridworld
from wurm.vis import plot_envs
from wurm.utils import env_consistency
from wurm.agents import A2C as Snake2C
from config import HEAD_CHANNEL, FOOD_CHANNEL


SEED = 543
GAMMA = 0.99
RENDER = False
LOG_INTERVAL = 100


parser = argparse.ArgumentParser()
parser.add_argument('--env')
parser.add_argument('--observation', default='default', type=str)
parser.add_argument('--coord-conv', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--num-envs', default=1, type=int)
parser.add_argument('--device', default='cpu', type=str)
args = parser.parse_args()


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
Transition = namedtuple('Transition', ['action', 'log_prob', 'value', 'reward', 'done'])


class A2C(nn.Module):
    def __init__(self, num_actions: int):
        super(A2C, self).__init__()
        self.num_actions = num_actions
        self.affine1 = nn.Linear(4, 128)
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
    size = 5
    env = SimpleGridworld(num_envs=args.num_envs, size=size, start_location=(2, 2), observation_mode=args.observation, device=args.device)
    if args.observation == 'positions':
        model = A2C(4).to(args.device)
    else:
        model = Snake2C(in_channels=2, size=size, coord_conv=args.coord_conv).to(args.device)
else:
    raise ValueError('Unrecognised environment')


def finish_episode(model, optimizer, saved_rewards, saved_actions):
    if len(saved_rewards) == 0 or len(saved_actions) == 0:
        return [], []

    R = 0
    policy_losses = []
    value_losses = []
    rewards = []
    for r in saved_rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)

    rewards = torch.stack(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.Tensor([r]).to(args.device)))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    if torch.isnan(loss):
        return [], []

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return [], []


optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()

running_reward = None
running_length = None
running_self_collisions = None
running_edge_collisions = None


saved_rewards = []
saved_actions = []
saved_transitions = []

episode_length = 0
num_episode = 0
num_steps = 0

rollouts = []

state = env.reset()
for i_step in count(1):
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample().clone().long()

    saved_actions.append(SavedAction(m.log_prob(action)[0], state_value[0]))

    state, reward, done, info = env.step(action)

    saved_rewards.append(reward[0])
    saved_transitions.append(Transition(action, m.log_prob(action), state_value, reward, done))

    env.reset(done)

    if done[0]:
        saved_rewards, saved_actions = finish_episode(model, optimizer, saved_rewards, saved_actions)
        episode_length = 0
        num_episode += 1
    else:
        episode_length += 1

    running_reward = reward.sum().item() if running_reward is None else running_reward * 0.975 + reward.sum().item() * 0.025
    running_length = episode_length if running_length is None else running_length * 0.975 + episode_length * 0.025
    if i_step % LOG_INTERVAL == 0:
        log_string = 'Steps {}\t'.format(i_step)
        log_string += 'Episode length: {:.2f}\t'.format(running_length)
        log_string += 'Episode reward: {:.2f}\t'.format(running_reward)
        print(log_string)
