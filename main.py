"""Entry point for training, transferring and visualising agents."""
from typing import List
import pandas as pd
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from wurm.envs import Slither, LaserTag
from wurm import agents
from wurm.envs.laser_tag.map_generators import MapFromString, MapPool
from wurm.callbacks.core import CallbackList
from wurm.callbacks.render import Render
from wurm.callbacks.model_checkpoint import ModelCheckpoint
from wurm.interaction import MultiSpeciesHandler
from wurm.rl import A2CLoss, A2CTrainer
from wurm.observations import FirstPersonCrop
from wurm import core
from wurm import arguments
from wurm import utils
from wurm.callbacks import loggers
from config import PATH, EPS


# This probably won't change so I've just made it a hardcoded value
INPUT_CHANNELS = 3


parser = argparse.ArgumentParser()
parser = arguments.add_common_arguments(parser)
parser = arguments.add_training_arguments(parser)
parser = arguments.add_model_arguments(parser)
parser = arguments.add_observation_arguments(parser)
parser = arguments.add_snake_env_arguments(parser)
parser = arguments.add_laser_tag_env_arguments(parser)
parser = arguments.add_render_arguments(parser)
parser = arguments.add_output_arguments(parser)
args = parser.parse_args()

included_args = ['env', 'n_envs', 'n_agents', 'repeat']

if args.laser_tag_map is not None:
    included_args += ['laser_tag_map', ]

argsdict = {k: v for k, v in args.__dict__.items() if k in included_args}
if 'agent' in argsdict.keys():
    argsdict['agent'] = argsdict['agent'][0]

    if 'agent=' in argsdict['agent']:
        argsdict['agent'] = argsdict['agent'].split('agent=')[1].split('__')[0]

if 'laser_tag_map' in argsdict.keys():
    argsdict['laser_tag_map'] = argsdict['laser_tag_map'][0]
argstring = '__'.join([f'{k}={v}' for k, v in argsdict.items()])

if args.dtype == 'float':
    dtype = torch.float
elif args.dtype == 'half':
    dtype = torch.half
else:
    raise RuntimeError

# Save file
if args.save_location is None:
    save_file = argstring
else:
    save_file = args.save_location

##########################
# Resume from checkpoint #
##########################
if os.path.exists(f'{PATH}/experiments/{args.save_folder}'):
    resume_from_checkpoint = True
    old_log_file = pd.read_csv(f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', comment='#')
    num_completed_steps = old_log_file.iloc[-1].steps
    num_completed_episodes = old_log_file.iloc[-1].episodes
    print('Pre-existing experiment folder detected - resuming from checkpoint')
    print(f'{num_completed_steps:.2e} out of {args.total_steps:.2e} environment steps completed.')
else:
    resume_from_checkpoint = False
    num_completed_steps = 0
    num_completed_episodes = 0

##########################
# Configure observations #
##########################
observation_function = FirstPersonCrop(
    height=args.obs_h,
    width=args.obs_w,
    first_person_rotation=args.obs_rotate,
    in_front=args.obs_in_front,
    behind=args.obs_behind,
    side=args.obs_side
)

#################
# Configure Env #
#################
render_args = {
    'size': args.render_window_size,
    'num_rows': args.render_rows,
    'num_cols': args.render_cols,
}
if args.env == 'snake':
    env = Slither(num_envs=args.n_envs, num_agents=args.n_agents, food_on_death_prob=args.food_on_death,
                  height=args.height, width=args.width, device=args.device, render_args=render_args, boost=args.boost,
                  boost_cost_prob=args.boost_cost, dtype=dtype, food_rate=args.food_rate,
                  respawn_mode=args.respawn_mode, food_mode=args.food_mode, observation_fn=observation_function,
                  reward_on_death=args.reward_on_death, agent_colours=args.colour_mode)
elif args.env == 'laser':
    if len(args.laser_tag_map) == 1:
        map_generator = MapFromString(args.laser_tag_map[0], args.device)
    else:
        fixed_maps = [MapFromString(m, args.device) for m in args.laser_tag_map]
        map_generator = MapPool(fixed_maps)

    env = LaserTag(num_envs=args.n_envs, num_agents=args.n_agents, height=args.height, width=args.width,
                   observation_fn=observation_function, colour_mode=args.colour_mode,
                   map_generator=map_generator, device=args.device, render_args=render_args)
elif args.env == 'cooperative':
    raise NotImplementedError
elif args.env == 'asymmetric':
    raise NotImplementedError
else:
    raise ValueError('Unrecognised environment')

num_actions = env.num_actions


###################
# Create agent(s) #
###################
num_heads = 1
num_models = args.n_species

# Quick hack to make reloading from checkpoint work
if resume_from_checkpoint:
    # Get latest checkpoint
    checkpoint_models = []
    for root, _, files in os.walk(f'{PATH}/experiments/{args.save_folder}/models/'):
        for f in sorted(files):
            model_args = {i.split('=')[0]: i.split('=')[1] for i in f[:-3].split('__')}
            checkpoint_models.append((int(model_args['species']), float(model_args['steps']), f))

    max_steps = max(checkpoint_models, key=lambda x: x[1])[1]

    latest_models = [os.path.join(root, f[2]) for f in checkpoint_models if f[1] == max_steps]
    args.agent_location = latest_models

# Quick hack to make it easier to input all of the species trained in one particular experiment
if args.agent_location is not None:
    if len(args.agent_location) == 1:
        species_0_path = args.agent_location[0]+'__species=0.pt'
        species_0_relative_path = os.path.join(PATH, args.agent_location[0])+'__species=0.pt'
        if os.path.exists(species_0_path) or os.path.exists(species_0_relative_path):
            agent_path = args.agent[0] if species_0_path else os.path.join(PATH, 'models', args.agent_location)
            args.agent_location = [agent_path + f'__species={i}.pt' for i in range(args.n_species)]
        else:
            args.agent_location = [args.agent_location[0], ] * args.n_agents

models: List[nn.Module] = []
for i in range(num_models):
    # Check for existence of model file
    if args.agent_location is None:
        specified_model_file = False
    else:
        model_path = args.agent_location[i]
        model_relative_path = os.path.join(PATH, 'models', args.agent_location[i])
        if os.path.exists(model_path) or os.path.exists(model_relative_path):
            specified_model_file = True
        else:
            specified_model_file = False

    # Create model class
    if args.agent_type == 'conv':
        models.append(
            agents.ConvAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=INPUT_CHANNELS, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64,
                num_heads=num_heads).to(device=args.device, dtype=dtype)
        )
    elif args.agent_type == 'gru':
        models.append(
            agents.GRUAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=INPUT_CHANNELS, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64,
                num_heads=num_heads).to(device=args.device, dtype=dtype)
        )
    elif args.agent_type == 'random':
        models.append(agents.RandomAgent(num_actions=num_actions, device=args.device))
    else:
        raise ValueError('Unrecognised agent type.')

    # Load state dict if the model file(s) have been specified
    if specified_model_file:
        print('Reloading agent {} from location {}'.format(i, args.agent_location[i]))
        models[i].load_state_dict(
            torch.load(args.agent_location[i])
        )

if args.train:
    for m in models:
        m.train()
else:
    torch.no_grad()
    for m in models:
        m.eval()
        for param in m.parameters():
            param.requires_grad = False

interaction_handler = MultiSpeciesHandler(models, args.n_species, args.n_agents, args.agent_type)
if args.train:
    a2c = A2CLoss(gamma=args.gamma, normalise_returns=args.norm_returns, dtype=dtype,
                  use_gae=args.gae_lambda is not None, gae_lambda=args.gae_lambda)
    a2c_trainer = A2CTrainer(
        models=models, update_steps=args.update_steps, lr=args.lr, a2c_loss=a2c, interaction_handler=interaction_handler,
        value_loss_coeff=args.value_loss_coeff, entropy_loss_coeff=args.entropy_loss_coeff, mask_dones=args.mask_dones,
        max_grad_norm=args.max_grad_norm)
else:
    a2c_trainer = None

callbacks = [
    loggers.LogEnricher(env, num_completed_steps, num_completed_episodes),
    loggers.PrintLogger(env=args.env, interval=args.print_interval),
    Render(env, args.fps) if args.render else None,
    ModelCheckpoint(
        f'{PATH}/experiments/{args.save_folder}/models',
        'steps={steps:.2e}__species={i_species}.pt',
        models,
        interval=args.model_interval
    ) if args.save_model else None,
    loggers.CSVLogger(
        filename=f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv',
        header_comment=utils.get_comment(args),
        interval=args.log_interval,
        append=resume_from_checkpoint
    ) if args.save_logs else None,
    loggers.VideoLogger(
        env,
        f'{PATH}/experiments/{args.save_folder}/videos/'
    ) if args.save_video else None,
    loggers.HeatMapLogger(
        env,
        save_folder=f'{PATH}/experiments/{args.save_folder}/heatmaps/',
        interval=args.heatmap_interval
    ) if args.save_heatmap else None

]
callbacks = [c for c in callbacks if c]
callback_list = CallbackList(callbacks)
callback_list.on_train_begin()

environment_run = core.EnvironmentRun(
    env=env,
    models=models,
    interaction_handler=interaction_handler,
    callbacks=callback_list,
    rl_trainer=a2c_trainer,
    warm_start=args.warm_start,
    initial_steps=num_completed_steps,
    total_steps=args.total_steps,
    initial_episodes=num_completed_episodes,
    total_episodes=args.total_episodes
)
environment_run.run()
