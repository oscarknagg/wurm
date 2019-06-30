from abc import ABC, abstractmethod
from typing import Dict, List
from torch import Tensor, nn
from torch.distributions import Categorical


class InteractionHandler(ABC):
    """Interface for interaction of multiple agents with an environment."""
    @abstractmethod
    def interact(self, observations: Dict[str, Tensor], cx: Dict[str, Tensor], hx: Dict[str, Tensor]) -> (Dict[str, Tensor], Dict[str, Tensor]):
        """


        Args:
            observations:
            cx:
            hx:

        Returns:
            action_probabilities:
            values:
        """
        raise NotImplementedError


class MultiSpeciesHandler(InteractionHandler):
    """Multiple species as models with unshared weights."""
    def __init__(self, models: List[nn.Module], n_species: int, n_agents: int, agent_type: str):
        assert len(models) == n_species
        self.models = models
        self.n_species = n_species
        self.n_agents = n_agents
        self.agent_type = agent_type

    def interact(self, observations: Dict[str, Tensor], cx: Dict[str, Tensor], hx: Dict[str, Tensor]) -> (Dict[str, Tensor], Dict[str, Tensor]):
        values = {}
        probs = {}
        for i, (agent, obs) in enumerate(observations.items()):
            model = self.models[i * self.n_species // self.n_agents]
            if self.agent_type == 'gru':
                probs_, value_, hx[agent] = model(obs, hx[agent])
            else:
                probs_, value_ = model(obs)

            probs[agent] = Categorical(probs_)
            values[agent] = value_

        return values, probs


class SharedBackboneSpeciesHandler(InteractionHandler):
    """Species share a common network with n_species policy and value heads."""
    def __init__(self, model: nn.Module, n_species: int, n_agents: int, agent_type: str):
        self.model = model
        self.n_species = n_species
        self.n_agents = n_agents
        self.agent_type = agent_type

    def interact(self, observations: Dict[str, Tensor], cx: Dict[str, Tensor], hx: Dict[str, Tensor]) -> (Dict[str, Tensor], Dict[str, Tensor]):
        values = {}
        probs = {}
        for i, (agent, obs) in enumerate(observations.items()):
            if self.agent_type == 'gru':
                probs_, value_, hx[agent] = self.model(obs, hx[agent])
            else:
                probs_, value_ = self.model(obs)

            # Select relevant policy and value outputs
            probs_ = probs_[i * self.n_species // self.n_agents]
            value_ = value_[i * self.n_species // self.n_agents]

            probs[agent] = Categorical(probs_)
            values[agent] = value_

        return values, probs


class ActionSampler(ABC):
    """Abstract for sampling method from probabilities."""
    @abstractmethod
    def sample(self, actions: Categorical):
        raise NotImplementedError


class StochasticSampler(ActionSampler):
    def sample(self, actions: Categorical):
        return actions.sample().clone().long()
