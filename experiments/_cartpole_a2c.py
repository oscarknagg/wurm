import gym
import numpy as np
from itertools import count
from collections import namedtuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from wurm.env import SingleSnakeEnvironments
from wurm.vis import plot_envs
from wurm.utils import env_consistency
from wurm.agents import A2C as Snake2C


SEED = 543
GAMMA = 0.99
RENDER = False
LOG_INTERVAL = 10
ENV = 'snake'


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


if ENV == 'cartpole':
    env = gym.make('CartPole-v0')
    env.seed(SEED)
    torch.manual_seed(SEED)
    model = A2C()
elif ENV == 'snake':
    size = 9
    env = SingleSnakeEnvironments(num_envs=1, size=size, device='cpu')
    model = Snake2C(in_channels=1, size=size).to('cpu')
else:
    raise ValueError('Unrecognised environment')


def select_action(model, state, action_store):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    # model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    action_store.append(SavedAction(m.log_prob(action), state_value))
    if ENV == 'cartpole':
        return action.item()
    elif ENV == 'snake':
        return torch.Tensor([action.item()]).long()
    else:
        raise ValueError('Unrecognised environment')


def finish_episode(model, optimizer, saved_rewards, saved_actions):
    # print('Finishing episode:')
    # print(len(saved_rewards), len(saved_actions))
    if len(saved_rewards) == 0 or len(saved_actions) == 0:
        return saved_rewards, saved_actions

    R = 0
    policy_losses = []
    value_losses = []
    rewards = []
    for r in saved_rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    if torch.isnan(loss):
        return rewards, saved_actions

    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()

    # del model.rewards[:]
    # del model.saved_actions[:]
    return [], []


optimizer = optim.Adam(model.parameters(), lr=3e-2 if ENV == 'cartpole' else 1e-3)
eps = np.finfo(np.float32).eps.item()

saved_rewards = []
saved_actions = []

running_reward = None
running_length = None
for i_episode in count(1):
    state = env.reset()
    for t in range(10000):  # Don't infinite loop while learning
        print(state[0, 0])
        print()
        action = select_action(model, state, saved_actions)
        state, reward, done, _ = env.step(action)

        if RENDER:
            env.render()

        saved_rewards.append(reward)

        if done:
            print(state[0, 0])
            exit()
            break

    episode_reward = sum(saved_rewards) if ENV == 'cartpole' else torch.stack(saved_rewards).sum().item()
    running_reward = episode_reward if running_reward is None else running_reward * 0.95 + episode_reward * 0.05
    running_length = t if running_length is None else running_length * 0.95 + t * 0.05
    saved_rewards, saved_actions = finish_episode(model, optimizer, saved_rewards, saved_actions)

    if i_episode % LOG_INTERVAL == 0:
        print('Episode {}\tAverage reward: {:.2f}\tEpisode length: {:.2f}'.format(
            i_episode, running_reward, running_length))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break

