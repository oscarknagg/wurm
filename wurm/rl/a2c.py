import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable

EPS = 1e-8


class A2C(object):
    """Class that encapsulates the advantage actor-critic algorithm.


    """
    def __init__(self,
                 actor_critic: nn.Module,
                 gamma: float,
                 value_loss_fn: Callable = F.smooth_l1_loss,
                 normalise_returns: bool = True):
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.normalise_returns = normalise_returns
        self.value_loss_fn = value_loss_fn

    def loss(self,
             bootstrap_values: torch.Tensor,
             rewards: torch.Tensor,
             values: torch.Tensor,
             log_probs: torch.Tensor,
             dones: torch.Tensor):
        # Only take whats absolutely necessary for A2C
        # Leave states behind
        # Leave entropy calculation to another piece of code
        R = bootstrap_values * (~dones[-1]).float()
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (~d).float()
            returns.insert(0, R)

        returns = torch.stack(returns)

        if self.normalise_returns:
            returns = (returns - returns.mean()) / (returns.std() + EPS)

        value_loss = self.value_loss_fn(values, returns).mean()
        advantages = returns - values
        policy_loss = - (advantages.detach() * log_probs).mean()

        return value_loss, policy_loss
