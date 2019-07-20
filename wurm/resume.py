from abc import ABC, abstractmethod
from argparse import Namespace
import pandas as pd
import os

from config import PATH


class Resume(ABC):
    """Abstract for classes that handle resuming experiments from a checkpoint."""
    @abstractmethod
    def resume(self, *args, **kwargs):
        raise NotImplementedError


class LocalResume(Resume):
    def resume(self, args: Namespace, save_file: str) -> (bool, int, int, dict):
        experiment_folder_exists = os.path.exists(f'{PATH}/experiments/{args.save_folder}')
        repeat_exists = os.path.exists(f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv')
        if experiment_folder_exists and repeat_exists:
            old_log_file = pd.read_csv(f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', comment='#')
            resume_from_checkpoint = True
            num_completed_steps = old_log_file.iloc[-1].steps
            num_completed_episodes = old_log_file.iloc[-1].episodes
            print('Pre-existing experiment folder detected - resuming from checkpoint')
            print(f'{num_completed_steps:.2e} out of {args.total_steps:.2e} environment steps completed.')
        else:
            resume_from_checkpoint = False
            num_completed_steps = 0
            num_completed_episodes = 0

        return resume_from_checkpoint, num_completed_steps, num_completed_episodes, {}


class S3Resume(Resume):
    def resume(self, *args, **kwargs):
        pass
