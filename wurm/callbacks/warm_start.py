from torch import nn
from typing import List
import torch

from wurm.core import MultiagentVecEnv
from wurm.interaction import InteractionHandler


class WarmStart:
    """Runs env for some steps before training starts."""
    def __init__(self, env: MultiagentVecEnv, models: List[nn.Module], num_steps: int, interaction_handler: InteractionHandler):
        super(WarmStart, self).__init__()
        self.env = env
        self.models = models
        self.num_steps = num_steps
        self.interaction_handler = interaction_handler

    def warm_start(self):
        # Run all agents for warm_start steps before training
        observations = self.env.reset()
        hidden_states = {f'agent_{i}': torch.zeros(
            (self.env.num_envs, 64), device=self.env.device) for i in range(self.env.num_agents)}
        cell_states = {f'agent_{i}': torch.zeros(
            (self.env.num_envs, 64), device=self.env.device) for i in range(self.env.num_agents)}

        for i in range(self.num_steps):
            interaction = self.interaction_handler.interact(observations, hidden_states, cell_states)

            observations, reward, done, info = self.env.step(interaction.actions)

            self.env.reset(done['__all__'])
            self.env.check_consistency()

            hidden_states = {k: v.detach() for k, v in hidden_states.items()}
            cell_states = {k: v.detach() for k, v in cell_states.items()}

        return observations, hidden_states, cell_states

