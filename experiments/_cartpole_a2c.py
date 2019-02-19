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

from wurm.env import SingleSnakeEnvironments
from wurm.vis import plot_envs
from wurm.utils import env_consistency
from wurm.agents import A2C as Snake2C
from config import HEAD_CHANNEL, FOOD_CHANNEL


SEED = 543
GAMMA = 0.99
RENDER = False
LOG_INTERVAL = 10


parser = argparse.ArgumentParser()
parser.add_argument('--env')
parser.add_argument('--observation', default='default', type=str)
parser.add_argument('--coord-conv', default=True, type=lambda x: x.lower()[0] == 't')
args = parser.parse_args()


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


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


if args.env == 'cartpole':
    env = gym.make('CartPole-v0')
    env.seed(SEED)
    torch.manual_seed(SEED)
    model = A2C(2)
elif args.env == 'snake':
    size = 12
    env = SingleSnakeEnvironments(num_envs=1, size=size, device='cpu', observation_mode=args.observation)
    if args.observation == 'positions':
        model = A2C(4)
    else:
        model = Snake2C(
            in_channels=1 if args.observation == 'one_channel' else 3, size=size, coord_conv=args.coord_conv).to('cpu')
else:
    raise ValueError('Unrecognised environment')


def select_action(model, state, action_store):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    # model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    action_store.append(SavedAction(m.log_prob(action), state_value))
    if args.env == 'cartpole':
        return action.item()
    elif args.env == 'snake':
        return torch.Tensor([action.item()]).long()
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

    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.Tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    if torch.isnan(loss):
        print('NAN')
        return [], []

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return [], []


optimizer = optim.Adam(model.parameters(), lr=3e-2 if args.env == 'cartpole' else 1e-3)
eps = np.finfo(np.float32).eps.item()
stay_alive_reward = 0.01

saved_rewards = []
saved_actions = []

running_reward = None
running_length = None
running_self_collisions = None
running_edge_collisions = None
for i_episode in count(1):
    episode_reward = []
    state = env.reset()
    for t in range(250):  # Don't infinite loop while learning
        action = select_action(model, state, saved_actions)
        state, reward, done, info = env.step(action)

        if RENDER:
            env.render()

        if args.env == 'snake':
            # Hacky reward shaping
            # head_idx = env.envs[:, HEAD_CHANNEL, :, :].view(env.num_envs, env.size ** 2).argmax(dim=-1)
            # food_idx = env.envs[:, FOOD_CHANNEL, :, :].view(env.num_envs, env.size ** 2).argmax(dim=-1)
            # head_pos = torch.Tensor((head_idx // env.size, head_idx % env.size)).float()
            # food_pos = torch.Tensor((food_idx // env.size, food_idx % env.size)).float()
            # food_closeness_reward = - torch.norm(head_pos - food_pos) * 0.01
            saved_rewards.append(reward + stay_alive_reward + done.float() * -10.)
        else:
            saved_rewards.append(reward)

        # Just intrinsic reward
        episode_reward.append(reward)

        if done:
            break

    saved_rewards, saved_actions = finish_episode(model, optimizer, saved_rewards, saved_actions)

    episode_reward = sum(episode_reward) if args.env == 'cartpole' else torch.stack(episode_reward).sum().item()
    running_reward = episode_reward if running_reward is None else running_reward * 0.95 + episode_reward * 0.05

    running_length = t if running_length is None else running_length * 0.95 + t * 0.05

    if args.env == 'snake':
        running_self_collisions = info['self_collision'].item() if running_self_collisions is None else running_self_collisions * 0.95 + info['self_collision'].item() * 0.05
        running_edge_collisions = info['edge_collision'].item() if running_edge_collisions is None else running_edge_collisions * 0.95 + info['edge_collision'].item() * 0.05

    if i_episode % LOG_INTERVAL == 0:
        log_string = 'Episode {}\tAverage reward: {:.2f}\tEpisode length: {:.2f}'.format(
            i_episode, running_reward, running_length)
        if args.env == 'snake':
            log_string += '\tEdge Collision: {:.1f}%'.format(running_edge_collisions * 100)
            log_string += '\tSelf Collision: {:.1f}%'.format(running_self_collisions * 100)
            log_string += '\tReward rate: {:.3f}'.format(running_reward / running_length)
        print(log_string)
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break

