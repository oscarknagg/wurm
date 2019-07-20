from abc import ABC, abstractmethod
from argparse import Namespace
from typing import NamedTuple, List
import pandas as pd
from botocore.exceptions import ClientError
import boto3
import os

from config import PATH


class ResumeData(NamedTuple):
    # This bool indicates whether or not we are resuming an experiment or
    # starting from scratch
    resume: bool
    current_steps: int
    current_episodes: int
    model_paths: List[str]


def build_resume_message(completed_steps: int, total_steps: int, completed_episodes: int, total_episodes: int,
                         log_file: str, model_paths: List[str]) -> str:
    msg = 'Resuming experiment:\n'
    msg += '-'*len('Resuming experiment:') + '\n'

    if total_steps < float('inf'):
        msg += '{} out {} steps completed.\n'.format(completed_steps, total_steps)

    if total_episodes < float('inf'):
        msg += '{} out {} steps completed.\n'.format(completed_episodes, total_episodes)

    msg += 'Pre-existing log file found at {}\n'.format(log_file)

    msg += 'Pre-existing model files found at [\n'

    for m in model_paths:
        msg += "    '{}',\n".format(m)

    msg += ']'

    return msg


class Resume(ABC):
    """Abstract for classes that handle resuming experiments from a checkpoint."""
    @staticmethod
    @abstractmethod
    def _resume_models(args: Namespace):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _resume_log(args: Namespace, save_file: str):
        raise NotImplementedError

    def resume(self, args: Namespace, save_file: str) -> ResumeData:
        logs_found, log_file, num_completed_steps, num_completed_episodes = self._resume_log(args, save_file)
        models_found, model_files = self._resume_models(args)

        # This operator is XOR. I know, I hadn't used this in python myself
        if logs_found ^ models_found:
            raise RuntimeError('Found logs but not models or vice versa.')
        elif logs_found and models_found:
            msg = build_resume_message(num_completed_steps, args.total_steps, num_completed_episodes,
                                       args.total_episodes, log_file, model_files)
            print(msg)
        else:
            print('Neither logs nor models found: starting experiment from scratch.')

        return ResumeData(
            logs_found and models_found,
            num_completed_steps,
            num_completed_episodes,
            model_files
        )


class LocalResume(Resume):
    @staticmethod
    def _resume_models(args: Namespace):
        if os.path.exists(f'{PATH}/experiments/{args.save_folder}/models/'):
            # Get latest checkpoint
            checkpoint_models = []
            for root, _, files in os.walk(f'{PATH}/experiments/{args.save_folder}/models/'):
                for f in sorted(files):
                    model_args = {i.split('=')[0]: i.split('=')[1] for i in f[:-3].split('__')}
                    checkpoint_models.append((int(model_args['species']), float(model_args['steps']), f))

            max_steps = max(checkpoint_models, key=lambda x: x[1])[1]

            latest_models = [os.path.join(root, f[2]) for f in checkpoint_models if f[1] == max_steps]
            args.agent_location = latest_models
            return bool(latest_models), latest_models
        else:
            return False, None

    @staticmethod
    def _resume_log( args: Namespace, save_file: str):
        experiment_folder_exists = os.path.exists(f'{PATH}/experiments/{args.save_folder}')
        repeat_exists = os.path.exists(f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv')
        if experiment_folder_exists and repeat_exists:
            old_log_file = pd.read_csv(f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', comment='#')
            logs_found = True
            num_completed_steps = old_log_file.iloc[-1].steps
            num_completed_episodes = old_log_file.iloc[-1].episodes
        else:
            logs_found = False
            num_completed_steps = 0
            num_completed_episodes = 0

        return logs_found, f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', num_completed_steps, num_completed_episodes


class S3Resume(Resume):
    @staticmethod
    def _resume_log(args: Namespace, save_file: str):
        # The easiest way to check if an experiment checkpoint exists is just to try to
        # load it and check for a 404.
        try:
            s3 = boto3.client('s3')
            s3.download_file(
                args.s3_bucket,
                f'{args.save_folder}/logs/{save_file}.csv',
                f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv'
            )
            old_log_file = pd.read_csv(f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', comment='#')

            logs_found = True
            num_completed_steps = old_log_file.iloc[-1].steps
            num_completed_episodes = old_log_file.iloc[-1].episodes
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                # Object not found
                logs_found = False
                num_completed_steps = 0
                num_completed_episodes = 0
            else:
                raise e
        except (FileNotFoundError, PermissionError) as e:
            # Another path for not finding the file
            # For some reason this raises a permission error as well
            logs_found = False
            num_completed_steps = 0
            num_completed_episodes = 0

        return logs_found, f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', num_completed_steps, num_completed_episodes

    @staticmethod
    def _resume_models(args: Namespace):
        s3 = boto3.client('s3')
        checkpoint_models = []
        object_query = s3.list_objects(Bucket=args.s3_bucket, Prefix=f'{args.save_folder}/models/')

        if 'Contents' in object_query:
            for key in object_query['Contents']:
                f = key['Key'].split('/')[-1]
                model_args = {i.split('=')[0]: i.split('=')[1] for i in f[:-3].split('__')}
                checkpoint_models.append((int(model_args['species']), float(model_args['steps']), f))

            max_steps = max(checkpoint_models, key=lambda x: x[1])[1]

            latest_models = [f's3://{args.s3_bucket}/{args.save_folder}/models/{f[2]}' for f in checkpoint_models if
                             f[1] == max_steps]

            return bool(latest_models), latest_models
        else:
            return False, None


"""
These two resume methods can be tested with the following commands:

# S3 resuming
# 1. Start very quick experiment from scratch
python main.py --env laser --n-envs 1 --n-agents 2 --n-species 2 --height 9 --width 9 \
    --total-steps 20 --agent-type gru --entropy 0.01 --lr 5e-4 --gamma 0.99 --mask-dones True \
    --obs-rotate True --obs-in-front 11 --obs-side 6 --obs-behind 2 \
    --laser-tag-map small2 --save-folder test --repeat 0 \
    --s3-bucket oscarknagg-experiments --s3-interval 10 --model-interval 10 --resume-mode s3

# 2. Run same experiment with a greater `total-steps`
python main.py --env laser --n-envs 1 --n-agents 2 --n-species 2 --height 9 --width 9 \
    --total-steps 40 --agent-type gru --entropy 0.01 --lr 5e-4 --gamma 0.99 --mask-dones True \
    --obs-rotate True --obs-in-front 11 --obs-side 6 --obs-behind 2 \
    --laser-tag-map small2 --save-folder test --repeat 0 \
    --s3-bucket oscarknagg-experiments --s3-interval 10 --model-interval 10 --resume-mode s3


# Local resuming
# 1. Start very quick experiment from scratch
python main.py --env laser --n-envs 1 --n-agents 2 --n-species 2 --height 9 --width 9 \
    --total-steps 20 --agent-type gru --entropy 0.01 --lr 5e-4 --gamma 0.99 --mask-dones True \
    --obs-rotate True --obs-in-front 11 --obs-side 6 --obs-behind 2 \
    --laser-tag-map small2 --save-folder test-local --repeat 0 \
    --model-interval 10 --resume-mode local

# 2. Run same experiment with a greater `total-steps`
python main.py --env laser --n-envs 1 --n-agents 2 --n-species 2 --height 9 --width 9 \
    --total-steps 40 --agent-type gru --entropy 0.01 --lr 5e-4 --gamma 0.99 --mask-dones True \
    --obs-rotate True --obs-in-front 11 --obs-side 6 --obs-behind 2 \
    --laser-tag-map small2 --save-folder test-local --repeat 0 \
    --model-interval 10 --resume-mode local

"""
