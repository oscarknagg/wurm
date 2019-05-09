from torch import nn
import torch.nn.functional as F
import torch

from wurm.modules import feedforward_block


class FeedforwardAgent(nn.Module):
    def __init__(self, num_actions: int,  num_layers: int, hidden_units: int, num_inputs: int = 4):
        super(FeedforwardAgent, self).__init__()
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.num_actions = num_actions

        feedforwards = [feedforward_block(num_inputs, self.hidden_units), ]
        for _ in range(self.num_layers - 1):
            feedforwards.append(feedforward_block(self.hidden_units, self.hidden_units))

        self.feedforward = nn.Sequential(*feedforwards)

        self.action_head = nn.Linear(self.hidden_units, self.num_actions)
        self.value_head = nn.Linear(self.hidden_units, 1)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.feedforward(x)
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values
