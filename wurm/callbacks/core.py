from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import torch
from torch.distributions import Distribution


class Callback(object):
    """Abstract class for callbacks"""
    def __init__(self):
        self.env = None

    def on_train_begin(self):
        pass

    def before_step(self, logs: Optional[dict] = None,
                    actions: Optional[Dict[str, torch.Tensor]] = None,
                    action_distributions: Optional[Dict[str, Distribution]] = None):
        pass

    def after_step(self,
                   logs: Optional[dict],
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        pass

    def on_train_end(self):
        pass


class CallbackList(object):
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def before_step(self, logs: Optional[dict] = None,
                    actions: Optional[Dict[str, torch.Tensor]] = None,
                    action_distributions: Optional[Dict[str, Distribution]] = None):
        for callback in self.callbacks:
            callback.before_step(logs, actions, action_distributions)

    def after_step(self,
                   logs: Optional[dict] = None,
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        for callback in self.callbacks:
            callback.after_step(logs, obs, rewards, dones, infos)

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

