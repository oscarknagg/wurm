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
from wurm.agents import A2C as Snake2C


SEED = 543
GAMMA = 0.99
RENDER = False
LOG_INTERVAL = 100
UPDATE_STEPS = 10
MAX_GRAD_NORM = 0.5


parser = argparse.ArgumentParser()
parser.add_argument('--env')
parser.add_argument('--observation', default='default', type=str)
parser.add_argument('--coord-conv', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--num-envs', default=1, type=int)
parser.add_argument('--verbose', default=0, type=int)
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
elif args.env == 'snake':
    size = 12
    env = SingleSnakeEnvironments(num_envs=args.num_envs, size=size, device=args.device, observation_mode=args.observation)
    if args.observation == 'positions':
        model = A2C(4).to(args.device)
    else:
        model = Snake2C(
            in_channels=1 if args.observation == 'one_channel' else 3, size=size, coord_conv=args.coord_conv).to(args.device)
else:
    raise ValueError('Unrecognised environment')


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
        running_length = episode_length if running_length is None else running_length * 0.975 + episode_length * 0.025
        episode_length = 0
        num_episode += 1
    else:
        episode_length += 1

    if i_step % UPDATE_STEPS == 0:
        if args.verbose > 1:
            print('========')

        with torch.no_grad():
            _, bootstrap_value = model(state)

        # R = bootstrap_value * (~done).float()
        R = 0
        returns = []
        for t in saved_transitions[::-1]:
            R = t.reward + GAMMA * R * (~t.done).float()
            returns.insert(0, R)

        returns = torch.stack(returns)

        # Normalise returns
        unnormalised_returns = returns.clone()
        # returns = (returns - returns.mean()) / (returns.std() + eps)
        # print('MINE:')
        # print(returns[:, 0])
        done = torch.stack([transition.done for transition in saved_transitions])
        rewards = torch.stack([transition.reward for transition in saved_transitions])

        policy_losses, value_losses = [], []
        for t, return_ in zip(saved_transitions, returns):
            advantage = return_ - t.value.detach()
            if args.verbose > 1:
                print(return_.item(), t.value.item(), advantage.item())
            policy_losses.append(-t.log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(t.value, return_))

        value_losses, policy_losses = torch.stack(value_losses), torch.stack(policy_losses)
        if args.verbose > 1:
            print('          return, done  , reward, v_loss, p_loss')
            print(torch.cat([
                unnormalised_returns.squeeze(-1),
                done.float().squeeze(-1),
                rewards.squeeze(-1),
                value_losses.unsqueeze(-1),
                policy_losses[:, 0]
            ], dim=-1))

        values = torch.stack([transition.value for transition in saved_transitions])
        value_loss = F.smooth_l1_loss(values, returns).mean()
        advantages = returns - values
        log_probs = torch.stack([transition.log_prob for transition in saved_transitions]).unsqueeze(-1)
        policy_loss = - (advantages.detach() * log_probs).mean()

        optimizer.zero_grad()
        loss = value_loss + policy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        saved_transitions = []

    running_reward = reward.mean().item() if running_reward is None else running_reward * 0.975 + reward.mean().item() * 0.025
    if args.env == 'snake':
        running_self_collisions = info[
            'self_collision'].float().mean().item() if running_self_collisions is None else running_self_collisions * 0.95 + info[
            'self_collision'].float().mean().item() * 0.05
        running_edge_collisions = info[
            'edge_collision'].float().mean().item() if running_edge_collisions is None else running_edge_collisions * 0.95 + info[
            'edge_collision'].float().mean().item() * 0.05

    if i_step % LOG_INTERVAL == 0:
        log_string = 'Steps {}\t'.format(i_step)
        log_string += 'Episode length: {:.3f}\t'.format(running_length)
        log_string += 'Episode reward: {:.3f}\t'.format(running_reward)

        if args.env == 'snake':
            log_string += '\tEdge Collision: {:.1f}%'.format(running_edge_collisions * 100)
            log_string += '\tSelf Collision: {:.1f}%'.format(running_self_collisions * 100)
            log_string += '\tReward rate: {:.3f}'.format(running_reward / running_length)
        print(log_string)
