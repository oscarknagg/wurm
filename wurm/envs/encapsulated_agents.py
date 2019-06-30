import torch
from torch import nn
from typing import Tuple, Dict, List
from torch.distributions import Categorical
from gym.envs.classic_control import rendering
import numpy as np
from PIL import Image

from .multi_snake import Slither
from config import DEFAULT_DEVICE


class AgentsAsEnvironment(object):
    """This env takes a multisnake env and some agents and makes it appear that its a single agent
    environment where the other agents are part of the environment (they don't learn).

    """
    def __init__(self,
                 num_envs: int,
                 num_npcs: int,
                 npc_agents: List[nn.Module],
                 npc_species: int,
                 npc_agent_type: str,
                 warm_start: int,
                 size: int,
                 initial_snake_length: int = 3,
                 on_death: str = 'restart',
                 observation_mode: str = 'full',
                 device: str = DEFAULT_DEVICE,
                 dtype: torch.dtype = torch.float,
                 manual_setup: bool = False,
                 food_on_death_prob: float = 0.5,
                 boost: bool = True,
                 boost_cost_prob: float = 0.5,
                 food_mode: str = 'only_one',
                 food_rate: float = 5e-4,
                 respawn_mode: str = 'all',
                 reward_on_death: int = -1,
                 verbose: int = 0,
                 render_args: dict = None,
                 agent_colours: str = 'random'):
        self._env = Slither(num_envs, num_npcs + 1, size, initial_snake_length, on_death, observation_mode,
                            device, dtype, manual_setup, food_on_death_prob, boost, boost_cost_prob,
                            food_mode, food_rate, respawn_mode, reward_on_death, verbose, render_args,
                            agent_colours)
        self.npc_agents = npc_agents
        self.num_npcs = num_npcs
        self.npc_species = npc_species
        self.npc_agent_type = npc_agent_type
        self.warm_start = warm_start

        self.observations = self._env.reset()
        self.npc_hidden_states = {
            f'agent_{i}': torch.zeros((self._env.num_envs, 64), device=self._env.device)
            for i in range(self._env.num_agents)
        }

        self.viewer = None

    def render(self, mode: str = 'human'):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        img = self.observations['agent_0']
        img = img.cpu().numpy()
        img = img[0]
        img = np.transpose(img, (1, 2, 0))

        img = np.array(Image.fromarray((img*255).astype(np.uint8)).resize(
            (self._env.render_args['size'] * 1,
             self._env.render_args['size'] * 1)
        ))

        if mode == 'human':
            # print(agent_obs.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow(agent_obs)
            # plt.show()
            # exit()
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return img
        else:
            raise ValueError('Render mode not recognised.')

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        all_actions = {}
        for i, (agent, obs) in enumerate(self.observations.items()):
            if i == 0:
                # Active agent
                all_actions[agent] = actions
            else:
                # Get the model for the corresponding agent
                model = self.npc_agents[(i - 1)//self.npc_species]
                if self.npc_agent_type == 'gru':
                    with torch.no_grad():
                        probs_, value_, self.npc_hidden_states[agent] = model(obs, self.npc_hidden_states.get(agent))
                else:
                    probs_, value_ = model(obs)

                action_distribution = Categorical(probs_)
                all_actions[agent] = action_distribution.sample().clone().long()

        self.observations, self.reward, self.done, self.info = self._env.step(all_actions)

        self._env.reset(self.done['__all__'])
        self._env.check_consistency()

        self.npc_hidden_states = {k: v.detach() for k, v in self.npc_hidden_states.items()}

        # Filter out the relevant obs, done, reward for the agent
        agent_obs = self.observations['agent_0']
        agent_reward = self.reward['agent_0']
        agent_done = self.done['agent_0']
        info = {}

        return agent_obs, agent_reward, agent_done, info

    def reset(self, *args, **kwargs) -> torch.Tensor:
        return self._env._observe()['agent_0']

    def check_consistency(self):
        self._env.check_consistency()
