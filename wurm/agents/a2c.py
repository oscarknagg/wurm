import torch
from torch import nn
import torch.nn.functional as F

from wurm.modules import CoordConv2D


class A2C(nn.Module):
    def __init__(self, in_channels: int, size: int, coord_conv: bool = True):
        """
        Advantage actor-critic (synchronous)

        For now use a hardcoded CNN policy

        Args:
            size: Size of environment
        """
        super(A2C, self).__init__()
        if coord_conv:
            self.conv = CoordConv2D(in_channels, 16, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.linear = nn.Linear(16 * size * size, 64)
        self.value_head = nn.Linear(64, 1)
        self.policy_head = nn.Linear(64, 4)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = F.relu(self.conv(x))
        x = F.relu(self.linear(x.view(x.shape[0], -1)))
        values = self.value_head(x)
        action_probabilities = self.policy_head(x)
        return F.softmax(action_probabilities, dim=-1), values
