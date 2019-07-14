import torch
from torch import nn
import torch.nn.functional as F

from wurm.agents.modules import CoordConv2D, ConvBlock, feedforward_block


class SimpleConvAgent(nn.Module):
    """
    Advantage actor-critic (synchronous)

    For now use a hardcoded CNN policy

    Args:
        size: Size of environment
    """
    def __init__(self, in_channels: int, size: int, coord_conv: bool = True, channels: int = 16):
        super(SimpleConvAgent, self).__init__()
        if coord_conv:
            self.conv1 = CoordConv2D(in_channels, channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1)

        self.linear = nn.Linear(channels * size * size, 64)
        self.value_head = nn.Linear(64, 1)
        self.policy_head = nn.Linear(64, 4)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.linear(x.view(x.shape[0], -1)))
        values = self.value_head(x)
        action_probabilities = self.policy_head(x)
        return F.softmax(action_probabilities, dim=-1), values


class ConvAgent(nn.Module):
    """Implementation of baseline agent architecture from https://arxiv.org/pdf/1806.01830.pdf"""
    def __init__(self,
                 in_channels: int,
                 num_initial_convs: int,
                 num_residual_convs: int,
                 num_feedforward: int,
                 feedforward_dim: int,
                 num_actions: int,
                 conv_channels: int = 16,
                 num_heads: int = 1):
        super(ConvAgent, self).__init__()
        self.in_channels = in_channels
        self.num_initial_convs = num_initial_convs
        self.num_residual_convs = num_residual_convs
        self.num_feedforward = num_feedforward
        self.feedforward_dim = feedforward_dim
        self.conv_channels = conv_channels
        self.num_actions = num_actions
        self.num_heads = num_heads

        initial_convs = [ConvBlock(self.in_channels, self.conv_channels, residual=False), ]
        for _ in range(self.num_initial_convs - 1):
            initial_convs.append(ConvBlock(self.conv_channels, self.conv_channels, residual=False))

        self.initial_conv_blocks = nn.Sequential(*initial_convs)

        residual_convs = [ConvBlock(self.conv_channels, self.conv_channels, residual=True), ]
        for _ in range(self.num_residual_convs - 1):
            residual_convs.append(ConvBlock(self.conv_channels, self.conv_channels, residual=True))

        self.residual_conv_blocks = nn.Sequential(*residual_convs)

        feedforwards = [feedforward_block(self.conv_channels, self.feedforward_dim), ]
        for _ in range(self.num_feedforward - 1):
            feedforwards.append(feedforward_block(self.feedforward_dim, self.feedforward_dim))

        self.feedforward = nn.Sequential(*feedforwards)

        self.value_head = nn.Linear(self.feedforward_dim, num_heads)
        self.policy_head = nn.Linear(self.feedforward_dim, self.num_actions * num_heads)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.initial_conv_blocks(x)
        x = self.residual_conv_blocks(x)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.feedforward(x)
        values = self.value_head(x)
        action_probabilities = self.policy_head(x)
        return F.softmax(action_probabilities, dim=-1), values
