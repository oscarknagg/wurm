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
from wurm.utils import env_consistency, CSVLogger
from config import BODY_CHANNEL, HEAD_CHANNEL, FOOD_CHANNEL, PATH


SEED = 543
RENDER = False
LOG_INTERVAL = 100
MAX_GRAD_NORM = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--env')
parser.add_argument('--observation', default='default', type=str)
parser.add_argument('--coord-conv', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--reward-shaping', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--num-envs', default=1, type=int)
parser.add_argument('--size', default=9, type=int)
parser.add_argument('--update-steps', default=20, type=int)
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--device', default='cpu', type=str)
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
    env = SimpleGridworld(num_envs=args.num_envs, size=size, start_location=(size//2, size//2), observation_mode=args.observation, device=args.device)
    if args.observation == 'positions':
        model = A2C(4).to(args.device)
    else:
        model = Snake2C(in_channels=2, size=size, coord_conv=args.coord_conv).to(args.device)
elif args.env == 'snake':
    size = args.size
    env = SingleSnakeEnvironments(num_envs=args.num_envs, size=size, device=args.device, observation_mode=args.observation)
    if args.observation == 'positions':
        model = A2C(4).to(args.device)
    if args.observation == 'partial':
        model = A2C(4, num_inputs=27).to(args.device)
    else:
        model = Snake2C(
            in_channels=1 if args.observation == 'one_channel' else 3, size=size, coord_conv=args.coord_conv).to(args.device)
else:
    raise ValueError('Unrecognised environment')


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

state = env.reset()
for i_step in count(1):
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

    if i_step % args.update_steps == 0:
        with torch.no_grad():
            _, bootstrap_value = model(state)

        R = bootstrap_value * (~done).float()
        returns = []
        for t in saved_transitions[::-1]:
            R = t.reward + args.gamma * R * (~t.done).float()
            returns.insert(0, R)

        returns = torch.stack(returns)

        # Normalise returns
        returns = (returns - returns.mean()) / (returns.std() + eps)
        done = torch.stack([transition.done for transition in saved_transitions])

        values = torch.stack([transition.value for transition in saved_transitions])
        value_loss = F.smooth_l1_loss(values, returns).mean()
        advantages = returns - values
        log_probs = torch.stack([transition.log_prob for transition in saved_transitions]).unsqueeze(-1)
        policy_loss = - (advantages.detach() * log_probs).mean()

        entropy_loss = - args.entropy * torch.stack([transition.entropy for transition in saved_transitions]).mean()

        optimizer.zero_grad()
        loss = value_loss + policy_loss + entropy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        saved_transitions = []

    # Metrics
    num_episodes += done.sum().item()
    num_steps += args.num_envs

    running_entropy = entropy.mean().item() if running_entropy is None else running_entropy * 0.975 + 0.025 * entropy.mean().item()
    running_reward_rate = reward.mean().item() if running_reward_rate is None else running_reward_rate * 0.975 + 0.025 * reward.mean().item()
    if args.env == 'snake':
        running_self_collisions = info[
            'self_collision'].float().mean().item() if running_self_collisions is None else running_self_collisions * 0.975 + info[
            'self_collision'].float().mean().item() * 0.025
        running_edge_collisions = info[
            'edge_collision'].float().mean().item() if running_edge_collisions is None else running_edge_collisions * 0.975 + info[
            'edge_collision'].float().mean().item() * 0.025

    if i_step % LOG_INTERVAL == 0:
        log_string = 'Steps {:.2f}e6\t'.format(num_steps/1e6)
        log_string += 'Reward rate: {:.3e}\t'.format(running_reward_rate)
        log_string += 'Entropy: {:.3e}\t'.format(running_entropy)

        logs = {
            'steps': num_steps,
            'episodes': num_episodes,
            'reward_rate': running_reward_rate,
            'policy_entropy': running_entropy,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
        }

        if args.env == 'snake':
            log_string += 'Avg. size: {:.3f}\t'.format(env.envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :].view(args.num_envs, -1).max(dim=1)[0].mean().item())
            log_string += 'Edge Collision: {:.3e}\t'.format(running_edge_collisions)
            log_string += 'Self Collision: {:.3e}\t'.format(running_self_collisions)
            # log_string += 'Reward rate: {:.3f}\tS'.format(running_reward / running_length)
            logs.update({
                'avg_size': env.envs[:, BODY_CHANNEL:BODY_CHANNEL + 1, :].view(args.num_envs, -1).max(dim=1)[0].mean().item(),
                'edge_collisions': running_edge_collisions,
                'self_collisions': running_self_collisions
            })
        logger.write(logs)
        print(log_string)

    if num_steps >= 100e6:
        break
