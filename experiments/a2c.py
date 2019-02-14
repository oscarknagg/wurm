import torch
import matplotlib.pyplot as plt

from wurm.env import SingleSnakeEnvironments
from wurm.vis import plot_envs
from wurm.agents import A2C
from wurm.utils import env_consistency
from config import DEFAULT_DEVICE


discount = 0.99
training_steps = 1000
size = 12
num_envs = 1
manual_setup = False


def sample_actions(action_probs):
    return torch.distributions.Categorical(action_probs).sample()


env = SingleSnakeEnvironments(num_envs=num_envs, size=size, manual_setup=manual_setup)
model = A2C(in_channels=3, size=size).to('cuda')
opt = torch.optim.Adam(A2C.parameters(), lr=1e-3)

observations = env.envs
for i in range(training_steps):
    action_probs, values = model(observations)

    actions = sample_actions(action_probs)

    observations, reward, done, info = env.step(actions)
    env.reset(done)
    env_consistency(env.envs)

    print(reward.sum().item())
    print()

    # Update model