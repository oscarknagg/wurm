from typing import Optional, Dict
from torch.distributions import Distribution
from time import sleep
import torch

from wurm.core import MultiagentVecEnv
from .core import Callback


class Render(Callback):
    def __init__(self, env: MultiagentVecEnv, fps: Optional[float] = 12):
        super(Render, self).__init__()
        self.env = env
        self.fps = fps

    def before_step(self, logs: Optional[dict] = None,
                    actions: Optional[Dict[str, torch.Tensor]] = None,
                    action_distributions: Optional[Dict[str, Distribution]] = None):
        self.env.render()
        sleep(1. / self.fps)
