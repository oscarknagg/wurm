from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
import numpy as np
import torch
from torch import nn
from PIL import Image
from gym.envs.classic_control import rendering
from itertools import count

from wurm.callbacks.core import CallbackList
from wurm.rl.core import RLTrainer
from wurm.interaction import InteractionHandler
from wurm import utils


class VecEnv(ABC):
    def __init__(self, num_envs: int, num_agents: int, height: int, width: int):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.height = height
        self.width = width

    @abstractmethod
    def step(self, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, dict):
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Optional[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def render(self, mode: str = 'human', env: Optional[int] = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def check_consistency(self):
        raise NotImplementedError


class MultiagentVecEnv(ABC):
    def __init__(self, num_envs: int,
                 num_agents: int,
                 height: int,
                 width: int,
                 dtype: torch.dtype,
                 device: str):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.height = height
        self.width = width
        self.dtype = dtype
        self.device = device
        self.viewer = None
        self.render_args = {'num_rows': 1, 'num_cols': 1, 'size': 256}

        # This Tensor represents the location of each agent in each environment. It should contain
        # only one non-zero entry for each sub array along dimension 0.
        self.agents = torch.zeros((num_envs * num_agents, 1, height, width), dtype=dtype, device=device, requires_grad=False)

        # This Tensor represents the current alive/dead state of each agent in each environment
        self.dones = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=device, requires_grad=False)

        # This tensor represents whether a particular environment has experienced an exception in the most recent
        # step. This is useful for resetting environments that have an exception
        self.errors = torch.zeros(self.num_envs, dtype=torch.uint8, device=device, requires_grad=False)

    @abstractmethod
    def step(self, actions: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict):
        raise NotImplementedError

    @abstractmethod
    def reset(self, done: torch.Tensor = None, return_observations: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def _get_env_images(self) -> torch.Tensor:
        """Gets RGB arrays for each environment.

        Returns:
            img: A Tensor of shape (num_envs, 3, height, width) and dtype torch.short i.e.
                an RBG rendering of each environment
        """
        raise NotImplementedError

    def render(self, mode: str = 'human', env: Optional[int] = None) -> Any:
        if self.viewer is None and mode == 'human':
            self.viewer = rendering.SimpleImageViewer(maxwidth=1080)

        img = self._get_env_images()
        img = build_render_rgb(img=img, num_envs=self.num_envs, env_height=self.height, env_width=self.width, env=env,
                               num_rows=self.render_args['num_rows'], num_cols=self.render_args['num_cols'],
                               render_size=self.render_args['size'])

        if mode == 'human':
            self.viewer.imshow(img)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return img
        else:
            raise ValueError('Render mode not recognised.')


def check_multi_vec_env_actions(actions: Dict[str, torch.Tensor], num_envs: int, num_agents: int):
    if len(actions) != num_agents:
        raise RuntimeError('Must have a Tensor of actions for each agent.')

    for agent, act in actions.items():
        if act.dtype not in (torch.short, torch.int, torch.long):
            raise TypeError('actions Tensor must be an integer type i.e. '
                            '{torch.ShortTensor, torch.IntTensor, torch.LongTensor}')

        if act.shape[0] != num_envs:
            raise RuntimeError('Must have the same number of actions as environments.')


class EnvironmentRun(object):
    def __init__(self,
                 env: MultiagentVecEnv,
                 models: List[nn.Module],
                 interaction_handler: InteractionHandler,
                 callbacks: CallbackList,
                 rl_trainer: Optional[RLTrainer],
                 warm_start: int = 0,
                 initial_steps: int = 0,
                 initial_episodes: int = 0,
                 total_steps: Optional[int] = None,
                 total_episodes: Optional[int] = None):
        self.env = env
        self.models = models
        self.interaction_handler = interaction_handler
        self.callbacks = callbacks
        self.rl_trainer = rl_trainer
        self.train = rl_trainer is not None
        self.warm_start = warm_start
        self.initial_steps = initial_steps
        self.initial_episodes = initial_episodes
        self.total_steps = total_steps if total_steps else float('inf')
        self.total_episodes = total_episodes if total_episodes else float('inf')

    def run(self):
        if self.warm_start:
            observations, hidden_states, cell_states = WarmStart(self.env, self.models, self.warm_start,
                                                                 self.interaction_handler).warm_start()
        else:
            observations = self.env.reset()
            hidden_states = {f'agent_{i}': torch.zeros((self.env.num_envs, 64), device=self.env.device) for i in
                             range(self.env.num_agents)}
            cell_states = {f'agent_{i}': torch.zeros((self.env.num_envs, 64), device=self.env.device) for i in
                           range(self.env.num_agents)}

        # Interact with environment
        num_episodes = self.initial_episodes
        num_steps = self.initial_steps
        final_logs = []
        for i_step in count(1):
            logs = {}

            interaction = self.interaction_handler.interact(observations, hidden_states, cell_states)
            self.callbacks.before_step(logs, interaction.actions, interaction.action_distributions)

            observations, reward, done, info = self.env.step(interaction.actions)
            self.env.reset(done['__all__'], return_observations=False)
            self.env.check_consistency()
            num_episodes += done['__all__'].sum().item()
            num_steps += self.env.num_envs

            with torch.no_grad():
                # Reset hidden states on death or on environment reset
                for _agent, _done in done.items():
                    if _agent != '__all__':
                        hidden_states[_agent][done['__all__']|_done].mul_(0)
                        cell_states[_agent][done['__all__']| _done].mul_(0)

            if not self.train:
                hidden_states = {k: v.detach() for k, v in hidden_states.items()}
                cell_states = {k: v.detach() for k, v in cell_states.items()}

            ##########################
            # Reinforcement learning #
            ##########################
            if self.train:
                self.rl_trainer.train(
                    interaction, hidden_states, cell_states, logs, observations, reward, done, info
                )

            self.callbacks.after_step(logs, observations, reward, done, info)

            final_logs.append(logs)

            if num_steps > self.total_steps or num_episodes >= self.total_episodes:
                break

        self.callbacks.on_train_end()

        return self.models, final_logs


def build_render_rgb(
        img: torch.Tensor,
        num_envs: int,
        env_height: int,
        env_width: int,
        num_rows: int,
        num_cols: int,
        render_size: int,
        env: Optional[int] = None) -> np.ndarray:
    """Util for viewing VecEnvs in a human friendly way.

    Args:
        img: Batch of RGB Tensors of the envs. Shape = (num_envs, 3, env_size, env_size).
        num_envs: Number of envs inside the VecEnv.
        env_height: Size of VecEnv.
        env_width: Size of VecEnv.
        num_rows: Number of rows of envs to view.
        num_cols: Number of columns of envs to view.
        render_size: Pixel size of each viewed env.
        env: Optional specified environment to view.
    """
    # Convert to numpy
    img = img.cpu().numpy()

    # Rearrange images depending on number of envs
    if num_envs == 1 or env is not None:
        num_cols = num_rows = 1
        img = img[env or 0]
        img = np.transpose(img, (1, 2, 0))
    else:
        num_rows = num_rows
        num_cols = num_cols
        # Make a grid of images
        output = np.zeros((env_height * num_rows, env_width * num_cols, 3))
        for i in range(num_rows):
            for j in range(num_cols):
                output[
                i * env_height:(i + 1) * env_height, j * env_width:(j + 1) * env_width, :
                ] = np.transpose(img[i * num_cols + j], (1, 2, 0))

        img = output

    ratio = env_width / env_height

    img = np.array(Image.fromarray(img.astype(np.uint8)).resize(
        (int(render_size * num_cols * ratio),
         int(render_size * num_rows))
    ))

    return img


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