import torch
from torch import nn
import torch.nn.functional as F

from wurm.modules import RelationalModule2D, CoordConv2D, AddCoords


def conv_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        CoordConv2D(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )


def feedforward_block(input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU()
    )


class RelationalBackbone(nn.Module):
    """Implementation of Relational agent architecture from https://arxiv.org/pdf/1806.01830.pdf"""

    def __init__(self,
                 num_convs: int,
                 in_channels: int,
                 conv_channels: int,
                 num_relational: int,
                 num_attention_heads: int,
                 relational_dim: int,
                 num_feedforward: int,
                 feedforward_dim: int,
                 residual: bool):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.num_relational = num_relational
        self.num_attention_heads = num_attention_heads
        self.relational_dim = relational_dim
        self.num_feedforward = num_feedforward
        self.feedforward_dim = feedforward_dim
        self.residual = residual

        convs = [conv_block(self.in_channels, self.conv_channels), ]
        for _ in range(self.num_convs-1):
            convs.append(conv_block(self.conv_channels, self.conv_channels))

        self.conv_blocks = nn.Sequential(*convs)
        self.coords = AddCoords()

        relational_blocks = [RelationalModule2D(self.num_attention_heads, self.conv_channels+2, self.relational_dim,
                                                residual=False)]
        for _ in range(self.num_relational-1):
            relational_blocks.append(RelationalModule2D(self.num_attention_heads, self.relational_dim,
                                                        self.relational_dim, residual=self.residual))

        self.relational_blocks = nn.Sequential(*relational_blocks)

        feedforwards = [feedforward_block(self.relational_dim, self.feedforward_dim), ]
        for _ in range(self.num_feedforward - 1):
            feedforwards.append(feedforward_block(self.feedforward_dim, self.feedforward_dim))

        self.feedforward = nn.Sequential(*feedforwards)

    def forward(self, x: torch.Tensor):
        x = self.conv_blocks(x)
        x = self.coords(x)
        x = self.relational_blocks(x)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.feedforward(x)
        return x
