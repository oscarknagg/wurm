"""Entry point for training, transferring and visualising agents."""
import numpy as np
from typing import List
from itertools import count
import argparse
from time import time, sleep
from pprint import pformat
import os
import git
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from wurm.envs import Slither, LaserTag
from wurm import agents
from wurm.callbacks.core import CallbackList
from wurm.callbacks.warm_start import WarmStart
from wurm.callbacks.render import Render
from wurm.callbacks.model_checkpoint import ModelCheckpoint
from wurm.interaction import MultiSpeciesHandler
from wurm.rl import A2CLoss, TrajectoryStore, A2CTrainer
from wurm.observations import FirstPersonCrop
from wurm import arguments
from wurm import utils
from wurm.callbacks import loggers
from config import PATH, EPS


parser = argparse.ArgumentParser()
parser = arguments.add_common_arguments(parser)
parser = arguments.add_training_arguments(parser)
parser = arguments.add_observation_arguments(parser)
parser = arguments.add_snake_env_arguments(parser)
parser = arguments.add_laser_tag_env_arguments(parser)
parser = arguments.add_render_arguments(parser)
parser = arguments.add_output_arguments(parser)
args = parser.parse_args()
callbacks = []

included_args = ['env', 'n_envs', 'n_agents', 'n_species', 'size', 'lr', 'gamma', 'update_steps', 'entropy', 'agent', 'obs',
                 'r']

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

in_channels = 3

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
    from wurm.envs.laser_tag.map_generators import MapFromString, MapPool

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
# If we share the network backbone between species then only create one network with n_species heads
if args.share_backbone:
    num_heads = args.n_species
    num_models = 1
else:
    num_heads = 1
    num_models = args.n_species

# Quick hack to make it easier to input all of the species trained in one particular experiment
if len(args.agent) == 1:
    species_0_path = args.agent[0]+'__species=0.pt'
    species_0_relative_path = os.path.join(PATH, 'models', args.agent[0])+'__species=0.pt'
    if os.path.exists(species_0_path) or os.path.exists(species_0_relative_path):
        agent_path = args.agent[0] if species_0_path else os.path.join(PATH, 'models', args.agent)
        args.agent = [agent_path + f'__species={i}.pt' for i in range(args.n_species)]
    else:
        args.agent = [args.agent[0], ] * args.n_agents

models: List[nn.Module] = []
for i in range(num_models):
    # Check for existence of model file
    model_path = args.agent[i]
    model_relative_path = os.path.join(PATH, 'models', args.agent[i])

    # Get agent params
    if os.path.exists(model_path) or os.path.exists(model_relative_path):
        agent_str = args.agent[i].split('/')[-1][:-3]
        agent_params = {kv.split('=')[0]: kv.split('=')[1] for kv in agent_str.split('__')}
        agent_type = agent_params['agent']
        reload = True
    else:
        agent_type = args.agent[i]
        reload = False

    # Create model class
    if agent_type == 'conv':
        models.append(
            agents.ConvAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64,
                num_heads=num_heads).to(device=args.device, dtype=dtype)
        )
    elif agent_type == 'gru':
        models.append(
            agents.GRUAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64,
                num_heads=num_heads).to(device=args.device, dtype=dtype)
        )
    elif agent_type == 'random':
        models.append(agents.RandomAgent(num_actions=num_actions, device=args.device))
    else:
        raise ValueError('Unrecognised agent type.')

    # Load state dict if reload
    if reload:
        print('Reloading agent {} from location {}'.format(i, args.agent[i]))
        models[i].load_state_dict(
            torch.load(args.agent[i])
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

if agent_type != 'random' and args.train:
    optimizers: List[optim.Adam] = []
    for i in range(num_models):
        optimizers.append(
            optim.Adam(models[i].parameters(), lr=args.lr, weight_decay=0)
        )

a2c = A2CLoss(gamma=args.gamma, normalise_returns=args.norm_returns, dtype=dtype,
          use_gae=args.gae_lambda is not None, gae_lambda=args.gae_lambda)
interaction_handler = MultiSpeciesHandler(models, args.n_species, args.n_agents, agent_type)
a2c_trainer = A2CTrainer(
    models=models, update_steps=args.update_steps, lr=args.lr, a2c_loss=a2c, interaction_handler=interaction_handler,
    value_loss_coeff=args.value_loss_coeff, entropy_loss_coeff=args.entropy_loss_coeff, mask_dones=args.mask_dones,
    max_grad_norm=args.max_grad_norm)


callbacks = [
    loggers.LogEnricher(env),
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
        interval=args.log_interval
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

trajectories = TrajectoryStore()
if agent_type != 'random' and args.train:
    optimizers: List[optim.Adam] = []
    for i in range(num_models):
        optimizers.append(
            optim.Adam(models[i].parameters(), lr=args.lr, weight_decay=0)
        )

##################
# Begin training #
##################
if args.warm_start:
    observations, hidden_states, cell_states = WarmStart(env, models, args.warm_start, interaction_handler).warm_start()
else:
    observations = env.reset()
    hidden_states = {f'agent_{i}': torch.zeros((args.n_envs, 64), device=args.device) for i in range(args.n_agents)}
    cell_states = {f'agent_{i}': torch.zeros((args.n_envs, 64), device=args.device) for i in range(args.n_agents)}

num_episodes = 0
num_steps = 0

for i_step in count(1):
    logs = {}

    interaction = interaction_handler.interact(observations, hidden_states, cell_states)

    callback_list.before_step(logs, interaction.actions, interaction.action_distributions)

    observations, reward, done, info = env.step(interaction.actions)
    env.reset(done['__all__'], return_observations=False)
    env.check_consistency()
    num_episodes += done['__all__'].sum().item()
    num_steps += args.n_envs

    if args.agent == 'gru' or args.agent == 'lstm':
        with torch.no_grad():
            # Reset hidden states on death or on environment reset
            for _agent, _done in done.items():
                if _agent != '__all__':
                    hidden_states[_agent | done['__all__']][_done].mul_(0)
                    cell_states[_agent | done['__all__']][_done].mul_(0)

    if args.agent != 'random' and args.train:
        a2c_trainer.train(
            interaction, hidden_states, cell_states, logs, observations, reward, done, info
        )

    callback_list.after_step(logs, observations, reward, done, info)

    if num_steps > args.total_steps or num_episodes >= args.total_episodes:
        break

callback_list.on_train_end()
