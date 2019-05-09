from torch import nn
import torch


class RandomAgent(nn.Module):
    def __init__(self, num_actions: int, device: str):
        super(RandomAgent, self).__init__()
        self.num_actions = num_actions
        self.device = device

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        n, c, h, w = x.size()
        return torch.ones((n, self.num_actions), device=self.device) / self.num_actions, torch.zeros(n)
