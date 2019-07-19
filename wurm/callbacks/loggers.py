import csv
import io
import os
from collections import OrderedDict
from typing import Iterable, Optional, Dict
import numpy as np
import torch
from torch.distributions import Distribution
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from time import time
import boto3

from wurm.utils import ExponentialMovingAverageTracker
from wurm.core import MultiagentVecEnv
from wurm import envs
from .core import Callback
from config import PATH, EPS


class CSVLogger(Callback):
    """Stream results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    Args:
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing training). False: overwrite existing file.
        header_comment: A possibly multi line string to put at the top of the CSV file
        """

    def __init__(self,
                 filename: str,
                 separator: str = ',',
                 append: bool = False,
                 header_comment: str = None,
                 interval: int = 1,
                 s3_bucket: Optional[str] = None,
                 s3_filename: Optional[str] = None,
                 s3_interval: int = 1000):
        super(CSVLogger, self).__init__()
        self.sep = separator
        self.filename = filename
        self.append = append
        self.header_comment = header_comment
        self.interval = interval
        self.s3_bucket = s3_bucket
        self.s3_filename = s3_filename
        self.s3_interval = s3_interval

        self.writer = None
        self.keys = None
        self.file_flags = ''
        self._open_args = {'newline': '\n'}

        # Make directory
        os.makedirs(os.path.split(filename)[0], exist_ok=True)

        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.write_header_comment = not bool(len(f.readline()))
                    self.write_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
            self.write_header_comment = True
            self.write_header = True

        self.write_header_comment = self.write_header_comment and self.header_comment is not None

        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

        self.i = 0

    def on_train_begin(self):
        def comment_out(s, comment='#'):
            return comment + s.replace('\n', f'\n{comment}')

        # Write the initial comment
        if self.write_header_comment:
            print(comment_out(self.header_comment), file=self.csv_file)

    def _write(self, logs: dict):
        def handle_value(k):
            is_zero_dim_tensor = isinstance(k, torch.Tensor) and k.ndimension() == 0
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray and not is_zero_dim_tensor:
                return '"[%s]"' % (', '.join(map(str, k)))
            elif is_zero_dim_tensor:
                return k.item()
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:
            # Write the column names
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=self.keys,
                                         dialect=CustomDialect)

            self.writer.writeheader()

        row_dict = OrderedDict()
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def after_step(self, logs: Optional[dict] = None,
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        if self.i % self.interval == 0:
            self._write(logs)

        if self.i % self.s3_interval and self.s3_bucket is not None:
            if self.s3_bucket is not None:
                boto3.client('s3').upload_file(
                    self.filename,
                    self.s3_bucket,
                    self.s3_filename
                )

        self.i += 1


class PrintLogger(Callback):
    def __init__(self, ewm_alpha: Optional[float] = 0.025, env: Optional[str] = None, interval: int = 1000):
        super(PrintLogger, self).__init__()
        self.ewm_alpha = ewm_alpha
        self.env = env
        self.interval = interval

        if self.ewm_alpha is not None:
            self.ewm_tracker = ExponentialMovingAverageTracker(alpha=0.025)

        self.i = 0

    def after_step(self, logs: Optional[dict] = None,
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        self.ewm_tracker(**logs)

        if self.i % self.interval == 0 and self.i > 0:
            log_string = '[{:02d}:{:02d}:{:02d}]\t'.format(
                int((logs['t'] // 3600)),
                int((logs['t'] // 60) % 60),
                int(logs['t'] % 60)
            )
            log_string += 'Steps {:.2f}e6\t'.format(logs['steps'] / 1e6)
            log_string += 'Entropy: {:.2e}\t'.format(self.ewm_tracker['policy_entropy_0'])
            log_string += 'Done: {:.2e}\t'.format(self.ewm_tracker['done_0'])
            log_string += 'Reward: {:.2e}\t'.format(self.ewm_tracker['reward_0'])

            if self.env == 'snake':
                log_string += 'Edge: {:.2e}\t'.format(self.ewm_tracker['edge_collisions_0'])
                log_string += 'Food: {:.2e}\t'.format(self.ewm_tracker['food_0'])
                log_string += 'Collision: {:.2e}\t'.format(self.ewm_tracker['snake_collisions_0'])
                log_string += 'Size: {:.3}\t'.format(self.ewm_tracker['size_0'])
                log_string += 'Boost: {:.2e}\t'.format(self.ewm_tracker['boost_0'])

            if self.env == 'laser':
                log_string += 'HP: {:.2e}\t'.format(self.ewm_tracker['hp_0'])
                log_string += 'Laser: {:.2e}\t'.format(self.ewm_tracker['laser_0'])

            log_string += 'FPS: {:.2e}\t'.format(self.ewm_tracker['fps'])

            print(log_string)

        self.i += 1


class LogEnricher(Callback):
    """Processes the logs immediately after a step/"""
    def __init__(self, env: MultiagentVecEnv, initial_steps: int = 0, initial_episodes: int = 0):
        super(LogEnricher, self).__init__()
        self.env = env

        self.num_episodes = initial_episodes
        self.num_steps = initial_steps

    def on_train_begin(self):
        self.t_train_begin = time()

    def before_step(self, logs: Optional[dict] = None,
                    actions: Optional[Dict[str, torch.Tensor]] = None,
                    action_distributions: Optional[Dict[str, Distribution]] = None):
        self.t_before_step = time()
        for i in range(self.env.num_agents):
            logs.update({
                f'policy_entropy_{i}': action_distributions[f'agent_{i}'].entropy().mean().item(),
            })

    def after_step(self, logs: Optional[dict] = None,
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        self.num_episodes += dones['__all__'].sum().item()
        self.num_steps += self.env.num_envs

        logs.update({
            't': time() - self.t_train_begin,
            'steps': self.num_steps,
            'episodes': self.num_episodes,
            'env_done': dones[f'__all__'].float().mean().item(),
            'fps': self.env.num_envs / (time() - self.t_before_step)
        })
        for i in range(self.env.num_agents):
            logs.update({
                f'reward_{i}': rewards[f'agent_{i}'].mean().item(),
                f'done_{i}': dones[f'agent_{i}'].float().mean().item(),
            })

            if isinstance(self.env, envs.Slither):
                logs.update({
                    f'boost_{i}': infos[f'boost_{i}'][~dones[f'agent_{i}']].float().mean().item(),
                    f'food_{i}': infos[f'food_{i}'][~dones[f'agent_{i}']].mean().item(),
                    f'edge_collisions_{i}': infos[f'edge_collision_{i}'].float().mean().item(),
                    f'snake_collisions_{i}': infos[f'snake_collision_{i}'].float().mean().item(),
                    f'size_{i}': infos[f'size_{i}'][~dones[f'agent_{i}']].mean().item(),
                    # Return of an episode is equivalent to the size on death
                    f'return_{i}': infos[f'size_{i}'][dones[f'agent_{i}']].mean().item(),
                })

            if isinstance(self.env, envs.LaserTag):

                logs.update({
                    f'hp_{i}': infos[f'hp_{i}'].float().mean().item(),
                    f'laser_{i}': infos[f'action_7_{i}'].float().mean().item(),
                    # f'hit_rate_{i}': 2 * rewards[f'agent_{i}'].mean().item() / infos[f'action_7_{i}'].float().mean().item(),
                    f'hit_rate_{i}': (infos[f'action_7_{i}'] & rewards[f'agent_{i}'].gt(EPS)).float().mean().item(),
                    f'errors': self.env.errors.sum().item()
                })


class VideoLogger(Callback):
    def __init__(self, env: MultiagentVecEnv, save_folder: str):
        super(VideoLogger, self).__init__()
        self.env = env
        self.save_folder = save_folder

        os.makedirs(save_folder, exist_ok=True)
        self.recorder = VideoRecorder(env, path=f'{save_folder}/0.mp4')

    def before_step(self, logs: Optional[dict] = None,
                    actions: Optional[Dict[str, torch.Tensor]] = None,
                    action_distributions: Optional[Dict[str, Distribution]] = None):
        self.recorder.capture_frame()

    def after_step(self, logs: Optional[dict] = None,
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        # If there is just one env save each episode to a different file
        # Otherwise save the whole video at the end
        if self.env.num_envs == 1:
            if logs['env_done']:
                # Save video and make a new recorder
                self.recorder.close()
                self.recorder = VideoRecorder(
                    self.env, path=f'{self.save_folder}/{logs["episodes"]}.mp4')

    def on_train_end(self):
        self.recorder.close()


class HeatMapLogger(Callback):
    """Logs a heatap of agent positions."""
    def __init__(self, env: MultiagentVecEnv, save_folder: str, interval: Optional[int] = 1000):
        super(HeatMapLogger, self).__init__()
        self.env = env
        self.save_folder = save_folder
        self.interval = interval

        os.makedirs(save_folder, exist_ok=True)

        self.agent_heatmap = torch.zeros((env.num_agents, env.height, env.width), device=env.device, dtype=torch.float,
                                         requires_grad=False)
        self.i = 0

    def after_step(self, logs: Optional[dict] = None,
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        self.agent_heatmap += self.env.agents\
            .view(self.env.num_envs, self.env.num_envs, self.env.height, self.env.width)\
            .sum(dim=0).clone()

        if self.i % self.interval == 0:
            os.makedirs(f'{PATH}/experiments/{self.save_folder}/heatmaps/', exist_ok=True)
            np.save(f'{PATH}/experiments/{self.save_folder}/heatmaps/{logs["steps"]}.npy', self.agent_heatmap.cpu().numpy())
            # Reset heatmap
            self.agent_heatmap = torch.zeros((self.env.num_agents, self.env.height, self.env.width),
                                             device=self.env.device, dtype=torch.float, requires_grad=False)

        self.i += 1
