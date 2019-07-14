import torch
from torch import nn
import torch.nn.functional as F

from wurm.agents.modules import RelationalModule2D, ConvBlock, feedforward_block


class RelationalAgent(nn.Module):
    """Implementation of Relational agent architecture from https://arxiv.org/pdf/1806.01830.pdf"""

    def __init__(self,
                 num_initial_convs: int,
                 in_channels: int,
                 conv_channels: int,
                 num_relational: int,
                 num_attention_heads: int,
                 relational_dim: int,
                 num_feedforward: int,
                 feedforward_dim: int,
                 residual: bool,
                 num_actions: int):
        super().__init__()
        self.num_initial_convs = num_initial_convs
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.num_relational = num_relational
        self.num_attention_heads = num_attention_heads
        self.relational_dim = relational_dim
        self.num_feedforward = num_feedforward
        self.feedforward_dim = feedforward_dim
        self.residual = residual
        self.num_actions = num_actions

        convs = [ConvBlock(self.in_channels, self.conv_channels, residual=False), ]
        for _ in range(self.num_initial_convs - 1):
            convs.append(ConvBlock(self.conv_channels, self.conv_channels, residual=False))

        self.initial_conv_blocks = nn.Sequential(*convs)

        relational_blocks = [RelationalModule2D(self.num_attention_heads, self.conv_channels, self.relational_dim,
                                                residual=False, add_coords=True)]
        for _ in range(self.num_relational-1):
            relational_blocks.append(
                RelationalModule2D(self.num_attention_heads, self.relational_dim,
                                   self.relational_dim, residual=self.residual, add_coords=True)
            )

        self.relational_blocks = nn.Sequential(*relational_blocks)

        feedforwards = [feedforward_block(self.relational_dim, self.feedforward_dim), ]
        for _ in range(self.num_feedforward - 1):
            feedforwards.append(feedforward_block(self.feedforward_dim, self.feedforward_dim))

        self.feedforward = nn.Sequential(*feedforwards)

        self.num_actions = num_actions
        self.action_head = nn.Linear(feedforward_dim, self.num_actions)
        self.value_head = nn.Linear(feedforward_dim, 1)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.initial_conv_blocks(x)
        x = self.relational_blocks(x)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.feedforward(x)
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values
