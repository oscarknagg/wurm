import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from wurm.env import SingleSnakeEnvironments
from wurm.vis import plot_envs
from wurm.agents import A2C
from wurm.utils import env_consistency
from config import DEFAULT_DEVICE


discount = 0.95
training_steps = 100000
size = 12
num_envs = 96
manual_setup = False


def sample_actions(action_probs):
    m = torch.distributions.Categorical(action_probs)
    a = m.sample()
    return a, m.log_prob(a)


env = SingleSnakeEnvironments(num_envs=num_envs, size=size, manual_setup=manual_setup)
model = A2C(in_channels=3, size=size).to('cuda')
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Initial step
observations = env.envs

print('step,reward,loss', file=open('../output.csv', 'w'))
# Ongoing steps
for i in range(training_steps):
    action_probs, values = model(observations.clone())
    actions, log_probs = sample_actions(action_probs)

    observations, rewards, done, info = env.step(actions)
    env.reset(done)
    env_consistency(env.envs)

    # TODO: store actions and rewards so I don't have to run model twice per iteration
    # with torch.no_grad():
    #     _, new_values = model(observations.clone())
    _, new_values = model(observations.clone())

    print('Rewards:', rewards.sum().item() / num_envs)

    # Update model
    opt.zero_grad()
    value_loss = F.smooth_l1_loss(values, rewards + discount * new_values)

    td_error = rewards.unsqueeze(-1) + discount * new_values - values
    policy_loss = - torch.distributions.Categorical(action_probs).log_prob(actions).unsqueeze(-1) * td_error
    entropies = -(action_probs.log() * action_probs)
    policy_loss -= entropies.sum(dim=-1, keepdim=True)
    loss = (policy_loss + value_loss).mean()
    # loss = policy_loss
    print(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    print('Entropies: ', entropies.mean())
    print('Actions: ', action_probs.mean(dim=0))
    print('Policy loss: ', policy_loss.mean().item())
    print('Value loss: ', value_loss.mean().item())
    print()

    # log_probs = new_log_probs
    # values = new_values

    print(f'{i},{rewards.sum().item() / num_envs},{loss.sum().item()}', file=open('../output.csv', 'a'))
