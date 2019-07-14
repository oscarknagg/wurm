import torch
from torch import nn
import torch.nn.functional as F

from wurm.agents.modules import ConvBlock, feedforward_block


class GRUAgent(nn.Module):
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
        super(GRUAgent, self).__init__()
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

        self.recurrent_module = nn.GRUCell(feedforward_dim, feedforward_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        x = self.initial_conv_blocks(x)
        x = self.residual_conv_blocks(x)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.feedforward(x)

        h = self.recurrent_module(x, h)

        values = self.value_head(h)
        action_probabilities = self.policy_head(h)
        if self.num_heads == 1:
            return F.softmax(action_probabilities, dim=-1), values, h
        else:
            return F.softmax(action_probabilities.view(-1, self.num_heads, self.num_actions), dim=-1), values, h
