import torch
from torch import nn, optim
from typing import Optional, Dict, List

from .a2c_loss import A2CLoss
from .core import RLTrainer
from .trajectory_store import TrajectoryStore
from wurm.interaction import InteractionHandler, Interaction
from wurm import utils


class A2CTrainer(RLTrainer):
    def __init__(self,
                 models: List[nn.Module],
                 update_steps: int,
                 lr: float,
                 a2c_loss: A2CLoss,
                 interaction_handler: InteractionHandler,
                 max_grad_norm: float,
                 value_loss_coeff: float,
                 entropy_loss_coeff: float,
                 mask_dones: bool = False):
        super(A2CTrainer, self).__init__()
        self.update_steps = update_steps
        self.models = models
        self.lr = lr
        self.interaction_handler = interaction_handler
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.mask_dones = mask_dones
        self.max_grad_norm = max_grad_norm

        self.optimizers = []
        for i, m in enumerate(models):
            self.optimizers.append(
                optim.Adam(m.parameters(), lr=lr, weight_decay=0)
            )
        self.trajectories = TrajectoryStore()
        self.a2c = a2c_loss

        self.i = 0

    def train(self,
              interaction: Interaction,
              hidden_states: Dict[str, torch.Tensor],
              cell_states: Dict[str, torch.Tensor],
              logs: Optional[dict],
              obs: Optional[Dict[str, torch.Tensor]] = None,
              rewards: Optional[Dict[str, torch.Tensor]] = None,
              dones: Optional[Dict[str, torch.Tensor]] = None,
              infos: Optional[Dict[str, torch.Tensor]] = None):
        self.trajectories.append(
            action=utils.stack_dict_of_tensors(interaction.actions),
            log_prob=utils.stack_dict_of_tensors(interaction.log_probs),
            value=utils.stack_dict_of_tensors(interaction.state_values),
            reward=utils.stack_dict_of_tensors(rewards),
            done=utils.stack_dict_of_tensors({k: v for k, v in dones.items() if k.startswith('agent_')}),
            entropy=utils.stack_dict_of_tensors(
                {k: v.entropy() for k, v in interaction.action_distributions.items()})
        )

        if self.i % self.update_steps == 0:
            with torch.no_grad():
                _interaction = self.interaction_handler.interact(obs, hidden_states, cell_states)
                bootstrap_values = utils.stack_dict_of_tensors(_interaction.state_values).detach()

            value_loss, policy_loss = self.a2c.loss(
                bootstrap_values.detach(),
                self.trajectories.rewards,
                self.trajectories.values,
                self.trajectories.log_probs,
                torch.zeros_like(self.trajectories.dones) if self.mask_dones else self.trajectories.dones
            )

            entropy_loss = - self.trajectories.entropies.mean()

            for opt in self.optimizers:
                opt.zero_grad()
            loss = self.value_loss_coeff * value_loss + policy_loss + self.entropy_loss_coeff * entropy_loss
            loss.backward()
            for model in self.models:
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            for opt in self.optimizers:
                opt.step()

            self.trajectories.clear()
            hidden_states = {k: v.detach() for k, v in hidden_states.items()}
            cell_states = {k: v.detach() for k, v in cell_states.items()}

        self.i += 1
