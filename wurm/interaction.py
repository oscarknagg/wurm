from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional
from torch import Tensor, nn
from torch.distributions import Categorical, Distribution


class Interaction(NamedTuple):
    """Representation of a single step of interaction between a set of agents and an environment."""
    action_distributions: Optional[Dict[str, Distribution]]
    actions: Optional[Dict[str, Tensor]]
    state_values: Optional[Dict[str, Tensor]]
    q_values: Optional[Dict[str, Tensor]]
    log_probs: Optional[Dict[str, Tensor]]


class InteractionHandler(ABC):
    """Interface for interaction of multiple agents with an environment."""
    @abstractmethod
    def interact(self, observations: Dict[str, Tensor], cx: Dict[str, Tensor], hx: Dict[str, Tensor]) -> Interaction:
        """


        Args:
            observations:
            cx:
            hx:

        Returns:
            interaction:
        """
        raise NotImplementedError


class ActionSampler(ABC):
    """Abstract for sampling method from probabilities."""
    @abstractmethod
    def sample(self, actions: Distribution) -> Tensor:
        raise NotImplementedError


class StochasticActionSampler(ActionSampler):
    def sample(self, actions: Distribution) -> Tensor:
        return actions.sample().clone().long()


class DeterministcActionSampler(ActionSampler):
    pass


class MultiSpeciesHandler(InteractionHandler):
    """Multiple species as models with unshared weights."""
    def __init__(self, models: List[nn.Module], n_species: int, n_agents: int, agent_type: str):
        assert len(models) == n_species
        self.models = models
        self.n_species = n_species
        self.n_agents = n_agents
        self.agent_type = agent_type

    def interact(self,
                 observations: Dict[str, Tensor],
                 hx: Optional[Dict[str, Tensor]],
                 cx: Optional[Dict[str, Tensor]]) -> Interaction:
        action_distributions = {}
        actions = {}
        values = {}
        log_probs = {}
        for i, (agent, obs) in enumerate(observations.items()):
            model = self.models[i * self.n_species // self.n_agents]
            if self.agent_type == 'lstm':
                probs_, value_, hx[agent], cx[agent] = model(obs, hx[agent], cx[agent])
            elif self.agent_type == 'gru':
                probs_, value_, hx[agent] = model(obs, hx[agent])
            else:
                probs_, value_ = model(obs)

            action_distributions[agent] = Categorical(probs_)
            actions[agent] = action_distributions[agent].sample().clone().long()
            values[agent] = value_
            log_probs[agent] = action_distributions[agent].log_prob(actions[agent].clone())

        interaction = Interaction(
            action_distributions=action_distributions,
            actions=actions,
            state_values=values,
            q_values=None,
            log_probs=log_probs
        )

        return interaction
