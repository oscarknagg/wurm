"""Takes a folder full of agents and calculates JPC."""
import os
import argparse
from itertools import product
from multiprocessing import Pool, cpu_count
import torch

from wurm.core import EnvironmentRun
from wurm.callbacks.core import CallbackList
from wurm.callbacks import loggers
from wurm.callbacks.render import Render
from wurm.interaction import MultiSpeciesHandler
from wurm import agents
from wurm import observations
from wurm import utils
from wurm import arguments
from config import PATH


INPUT_CHANNELS = 3


def get_bool(input_string):
    return input_string.lower()[0] == 't'


parser = argparse.ArgumentParser()
parser.add_argument('--experiment-folder', type=str)
parser.add_argument('--steps', type=float)
parser.add_argument('--processes', type=int, default=cpu_count())
parser.add_argument('--agent-type', type=str)
parser.add_argument('--save-video', type=get_bool)
parser = arguments.add_common_arguments(parser)
parser = arguments.add_laser_tag_env_arguments(parser)
parser = arguments.add_snake_env_arguments(parser)
parser = arguments.add_observation_arguments(parser)
parser = arguments.add_render_arguments(parser)

args = parser.parse_args()
model_folder = os.path.join('experiments', args.experiment_folder, 'models')

if args.dtype == 'float':
    args.dtype = torch.float
elif args.dtype == 'half':
    args.dtype = torch.half
else:
    raise RuntimeError

models_to_run = {}
for root, _, files in os.walk(model_folder):
    for f in files:
        model_args = {i.split('=')[0]: i.split('=')[1] for i in f[:-3].split('__')}
        if float(model_args['steps']) == args.steps:
            models_to_run.update({
                (int(model_args['repeat']), int(model_args['species'])): os.path.join(root, f)
            })

n_repeats = max({k[0] for k in models_to_run.keys()}) + 1
n_species = max({k[1] for k in models_to_run.keys()}) + 1

##########################
# Configure observations #
##########################
observation_function = observations.FirstPersonCrop(
    height=args.obs_h,
    width=args.obs_w,
    first_person_rotation=args.obs_rotate,
    in_front=args.obs_in_front,
    behind=args.obs_behind,
    side=args.obs_side
)


def worker(i, j):
    env = arguments.get_env(args)
    env.observation_fn = observation_function

    model_locations = [
        models_to_run[i, 0],
        models_to_run[j, 1]
    ]
    models = []
    for model_path in model_locations:
        models.append(
            agents.GRUAgent(
                num_actions=env.num_actions, num_initial_convs=2, in_channels=INPUT_CHANNELS, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=args.device, dtype=args.dtype)
        )
        models[-1].load_state_dict(torch.load(model_path))

    interaction_handler = MultiSpeciesHandler(models, args.n_species, args.n_agents, args.agent_type)
    print(f'{PATH}/experiments/{args.experiment_folder}/jpc/{i}-{j}.csv')
    callbacks = [
        loggers.LogEnricher(env),
        Render(env, args.fps) if args.render else None,
        loggers.CSVLogger(
            filename=f'{PATH}/experiments/{args.experiment_folder}/jpc/{args.steps}/{i}-{j}.csv',
        ),
        loggers.VideoLogger(
            env,
            f'{PATH}/experiments/{args.experiment_folder}/videos/jpc/{args.steps}/{i}-{j}/'
        ) if args.save_video else None,

    ]
    callbacks = [c for c in callbacks if c]
    callback_list = CallbackList(callbacks)
    callback_list.on_train_begin()

    environment_run = EnvironmentRun(
        env=env,
        models=models,
        interaction_handler=interaction_handler,
        callbacks=callback_list,
        rl_trainer=None,
        total_episodes=args.n_envs,
    )
    environment_run.run()


indices = list(product(range(n_repeats), range(n_repeats)))

pool = Pool(args.processes)
try:
    pool.starmap(worker, indices)
except KeyboardInterrupt:
    raise Exception
finally:
    pool.close()
