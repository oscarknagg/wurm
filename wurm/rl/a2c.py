import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable

EPS = 1e-8


class A2C(object):
    """Class that encapsulates the advantage actor-critic algorithm.

    Args:
        actor_critic: Module that outputs
        gamma: Discount value
        value_loss_fn: Loss function between values and returns i.e. Huber, MSE
        normalise_returns: Whether or not to normalise target returns
    """
    def __init__(self,
                 gamma: float,
                 value_loss_fn: Callable = F.smooth_l1_loss,
                 normalise_returns: bool = False,
                 use_gae: bool = False,
                 gae_lambda: float = None,
                 dtype: torch.dtype = torch.float):
        self.gamma = gamma
        self.normalise_returns = normalise_returns
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.value_loss_fn = value_loss_fn
        self.dtype = dtype

    def loss(self,
             bootstrap_values: torch.Tensor,
             rewards: torch.Tensor,
             values: torch.Tensor,
             log_probs: torch.Tensor,
             dones: torch.Tensor):
        """Calculate A2C loss.

        Args:
            bootstrap_values: Vector containing estimated value of final states each trajectory.
                Shape (num_envs, 1)
            rewards: Rewards for trajectories. Shape: (num_envs, num_steps)
            values: Values for trajectory states: Shape (num_envs, num_steps)
            log_probs: Log probabilities of actions taken during trajectory. Shape: (num_envs, num_steps)
            dones: Done masks for trajectory states. Shape: (num_envs, num_steps)
        """
        returns = []
        if self.use_gae:
            gae = 0
            for t in reversed(range(rewards.size(0))):
                if t == rewards.size(0) - 1:
                    delta = rewards[t] + self.gamma * bootstrap_values * (~dones[t]).to(self.dtype) - values[t]
                else:
                    delta = rewards[t] + self.gamma * values[t+1] * (~dones[t]).to(self.dtype) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (~dones[t]).to(self.dtype) * gae
                R = gae + values[t]
                returns.insert(0, R)
        else:
            R = bootstrap_values * (~dones[-1]).to(self.dtype)
            for r, d in zip(reversed(rewards), reversed(dones)):
                R = r + self.gamma * R * (~d).to(self.dtype)
                returns.insert(0, R)

        returns = torch.stack(returns)

        if self.normalise_returns:
            returns = (returns - returns.mean()) / (returns.std() + EPS)

        value_loss = self.value_loss_fn(values, returns).mean()
        advantages = returns - values
        policy_loss = - (advantages.detach() * log_probs).mean()

        return value_loss, policy_loss
