import torch
from torch import nn
import torch.nn.functional as F

from wurm.modules import CoordConv2D, ConvBlock, feedforward_block


class ConvDiscriminator(nn.Module):
    """Implementation of baseline agent architecture from https://arxiv.org/pdf/1806.01830.pdf"""
    def __init__(self,
                 in_channels: int,
                 num_initial_convs: int,
                 num_residual_convs: int,
                 num_feedforward: int,
                 feedforward_dim: int,
                 num_species: int,
                 conv_channels: int = 16):
        super(ConvDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.num_initial_convs = num_initial_convs
        self.num_residual_convs = num_residual_convs
        self.num_feedforward = num_feedforward
        self.feedforward_dim = feedforward_dim
        self.conv_channels = conv_channels
        self.num_species = num_species

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

        self.species_head = nn.Linear(self.feedforward_dim, self.num_species)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.initial_conv_blocks(x)
        x = self.residual_conv_blocks(x)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.feedforward(x)
        species_logits = self.species_head(x)
        return species_logits
