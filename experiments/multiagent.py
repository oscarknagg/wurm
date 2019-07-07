"""Entry point for training, transferring and visualising agents."""
import numpy as np
from typing import List
from itertools import count
import argparse
from time import time, sleep
from pprint import pprint, pformat
import os
import git
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from wurm.envs import Slither, LaserTag
from wurm import agents
from wurm.utils import CSVLogger, ExponentialMovingAverageTracker
from wurm.rl import A2C, TrajectoryStore
from wurm.agents.discriminator import ConvDiscriminator
from wurm.observations import FirstPersonCrop
from config import PATH, EPS


def flatten_dict(d, take_every=1):
    flattened = []
    for i, (k, v) in enumerate(d.items()):
        if i % take_every == 0:
            flattened.append(v)

    flattened = torch.stack(flattened).view(-1, 1)
    return flattened


def get_bool(input_string):
    return input_string.lower()[0] == 't'


VALUE_LOSS_COEFF = 0.5
LOG_INTERVAL = 1
MODEL_INTERVAL = 1000
PRINT_INTERVAL = 1000
HEATMAP_INTERVAL = 1000
MAX_GRAD_NORM = 0.5
FPS = 10


parser = argparse.ArgumentParser()

# Generic arguments
parser.add_argument('--env', type=str)
parser.add_argument('--n-envs', type=int)
parser.add_argument('--n-agents', type=int)
parser.add_argument('--n-species', type=int, default=1)
parser.add_argument('--height', type=int)
parser.add_argument('--width', type=int)
parser.add_argument('--agent', type=str, nargs='+')
parser.add_argument('--coord-conv', default=True, type=get_bool)
parser.add_argument('--mask-dones', default=True, type=get_bool, help='Removes deaths from training trajectories.')
parser.add_argument('--warm-start', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dtype', type=str, default='float')
parser.add_argument('--diayn', default=0, type=float)
parser.add_argument('--r', default=None, type=int, help='Repeat number')

# Training arguments
parser.add_argument('--train', default=True, type=get_bool)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--gae-lambda', default=None, type=float)
parser.add_argument('--update-steps', default=5, type=int)
parser.add_argument('--entropy', default=0.0, type=float)
parser.add_argument('--entropy-min', default=None, type=float)
parser.add_argument('--total-steps', default=float('inf'), type=float)
parser.add_argument('--total-episodes', default=float('inf'), type=float)
parser.add_argument('--norm-returns', default=False, type=get_bool)
parser.add_argument('--share-backbone', default=False, type=get_bool)

# Observation arguments
parser.add_argument('--obs-h', type=int)
parser.add_argument('--obs-w', type=int)
parser.add_argument('--obs-rotate', type=get_bool)
parser.add_argument('--obs-in-front', type=int)
parser.add_argument('--obs-behind', type=int)
parser.add_argument('--obs-side', type=int)

# Snake arguments
parser.add_argument('--boost', default=True, type=get_bool)
parser.add_argument('--boost-cost', type=float, default=0.25)
parser.add_argument('--food-on-death', type=float, default=0.33)
parser.add_argument('--food-on-death-min', type=float, default=None)
parser.add_argument('--reward-on-death', type=float, default=-1)
parser.add_argument('--food-mode', type=str, default='random_rate')
parser.add_argument('--food-rate', type=float, default=3e-4)
parser.add_argument('--food-rate-min', type=float, default=None)
parser.add_argument('--respawn-mode', type=str, default='any')
parser.add_argument('--colour-mode', type=str, default='random')

# Laser tag arguments
parser.add_argument('--laser-tag-map', type=str, default='random')

# Render arguments
parser.add_argument('--render', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--render-window-size', default=256, type=int)
parser.add_argument('--render-cols', default=1, type=int)
parser.add_argument('--render-rows', default=1, type=int)

# Output arguments
parser.add_argument('--save-folder', type=str, default=None)
parser.add_argument('--save-location', type=str, default=None)
parser.add_argument('--save-model', default=True, type=get_bool)
parser.add_argument('--save-logs', default=True, type=get_bool)
parser.add_argument('--save-video', default=False, type=get_bool)
parser.add_argument('--save-heatmap', default=False, type=get_bool)

args = parser.parse_args()

included_args = ['env', 'n_envs', 'n_agents', 'n_species', 'size', 'lr', 'gamma', 'update_steps', 'entropy', 'agent', 'obs',
                 'r']

if args.laser_tag_map is not None:
    included_args += ['laser_tag_map', ]

entropy_coeff = args.entropy

argsdict = {k: v for k, v in args.__dict__.items() if k in included_args}
if 'agent' in argsdict.keys():
    argsdict['agent'] = argsdict['agent'][0]
argstring = '__'.join([f'{k}={v}' for k, v in argsdict.items()])
print(argstring)

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

if args.save_folder is not None:
    save_file = os.path.join(args.save_folder, save_file)

# In channels
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
    from wurm.envs.laser_tag.map_generators import MapFromString
    from wurm.envs.laser_tag.maps import small2, small3, small4
    if args.laser_tag_map == 'small2':
        map_generator = MapFromString(small2, args.device)
    elif args.laser_tag_map == 'small3':
        map_generator = MapFromString(small3, args.device)
    elif args.laser_tag_map == 'small4':
        map_generator = MapFromString(small4, args.device)
    else:
        raise ValueError('Unrecognised LaserTag map')

    env = LaserTag(num_envs=args.n_envs, num_agents=args.n_agents, height=args.height, width=args.width,
                   observation_fn=observation_function, colour_mode=args.colour_mode,
                   map_generator=map_generator, device=args.device, render_args=render_args)
elif args.env == 'tron':
    raise NotImplementedError
elif args.env == 'bomb':
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
        # observation_type = agent_params['obs']
        reload = True
    else:
        agent_type = args.agent[i]
        # observation_type = args.obs
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

#########
# DIAYN #
#########
if args.diayn > 0:
    discriminator = ConvDiscriminator(
        num_species=args.n_species, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
        num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=args.device, dtype=dtype)
    discrim_opt = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=1e-5)


trajectories = TrajectoryStore()
ewm_tracker = ExponentialMovingAverageTracker(alpha=0.025)

episode_length = 0
num_episodes = 0
num_steps = 0
if args.save_logs:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    comment = f'Git commit: {sha}\n'
    comment += f'Args: {json.dumps(args.__dict__)}\n'
    comment += 'Prettier args:\n'
    comment += pformat(args.__dict__)
    logger = CSVLogger(filename=f'{PATH}/logs/{save_file}.csv', header_comment=comment)
if args.save_video:
    os.makedirs(PATH + f'/videos/{save_file}', exist_ok=True)
    recorder = VideoRecorder(env, path=PATH + f'/videos/{save_file}/0.mp4')

a2c = A2C(gamma=args.gamma, normalise_returns=args.norm_returns, dtype=dtype,
          use_gae=args.gae_lambda is not None, gae_lambda=args.gae_lambda)


############################
# Run agent in environment #
############################
t0 = time()
hidden_states = {f'agent_{i}': torch.zeros((args.n_envs, 64), device=args.device) for i in range(args.n_agents)}
if args.warm_start:
    # Run all agents for warm_start steps before training
    observations = env.reset()
    for i in range(args.warm_start):
        actions = {}
        for i, (agent, obs)in enumerate(observations.items()):
            # Get the model for the corresponding species
            species_idx = i * args.n_species // args.n_agents
            model = models[0 if args.share_backbone else species_idx]
            if agent_type == 'gru':
                probs_, value_, hidden_states[agent] = model(obs, hidden_states.get(agent))
            else:
                probs_, value_ = model(obs)

            if args.share_backbone:
                # In this case we need to select the correct head of the network
                probs_ = probs_[:, species_idx]
                value_ = value_[:, species_idx]

            action_distribution = Categorical(probs_)
            actions[agent] = action_distribution.sample().clone().long()

        observations, reward, done, info = env.step(actions)

        env.reset(done['__all__'])
        env.check_consistency()

        if not args.train:
            hidden_states = {k: v.detach() for k, v in hidden_states.items()}

if not args.warm_start:
    observations = env.reset()

if args.save_heatmap:
    head_heatmap = torch.zeros((args.n_agents, args.size, args.size), device=args.device, dtype=torch.float, requires_grad=False)

for i_step in count(1):
    logs = {}
    t_i = time()
    if args.render:
        env.render()
        sleep(1. / FPS)

    if args.save_video:
        recorder.capture_frame()

    ########################
    # Hyperparam annealing #
    ########################
    if args.entropy_min is not None:
        # Per tick entropy annealing
        entropy_delta = (args.entropy - args.entropy_min) / args.total_steps
        entropy_coeff -= entropy_delta * args.n_envs

    if args.food_rate_min is not None:
        # Per tick entropy annealing
        food_delta = (args.food_rate - args.food_rate_min) / args.total_steps
        env.food_rate -= food_delta * args.n_envs

    if args.food_on_death_min is not None:
        # Per tick entropy annealing
        food_on_death_delta = (args.food_on_death - args.food_on_death_min) / args.total_steps
        env.food_on_death_prob -= food_on_death_delta * args.n_envs

    #############################
    # Interact with environment #
    #############################
    actions = {}
    values = {}
    probs = {}
    entropies = {}
    for i, (agent, obs) in enumerate(observations.items()):
        # Get the model for the corresponding species
        species_idx = i * args.n_species // args.n_agents
        model = models[0 if args.share_backbone else species_idx]
        if agent_type == 'gru':
            probs_, value_, hidden_states[agent] = model(obs, hidden_states.get(agent))
        else:
            probs_, value_ = model(obs)

        if args.share_backbone:
            # In this case we need to select the correct head of the network
            # print(i_step, i, species_idx, probs_.shape)
            probs_ = probs_[:, species_idx]
            value_ = value_[:, species_idx]

        action_distribution = Categorical(probs_)
        actions[agent] = action_distribution.sample().clone().long()
        values[agent] = value_
        entropies[agent] = action_distribution.entropy()
        probs[agent] = action_distribution.log_prob(actions[agent].clone())

    observations, reward, done, info = env.step(actions)

    env.reset(done['__all__'], return_observations=False)
    env.check_consistency()

    if args.diayn > 0:
        # DIAYN
        discrim_opt.zero_grad()
        discrim_loss = 0
        for i, (agent, obs) in enumerate(observations.items()):
            species_idx = i * args.n_species // args.n_agents
            species_label = torch.tensor([species_idx, ]*args.n_envs,
                                         device=args.device, dtype=torch.long, requires_grad=False)
            # Optimise discriminator
            species_predictions = discriminator(observations[agent])
            discrim = F.cross_entropy(species_predictions, species_label, reduction='none')
            # Add pseudo reward
            with torch.no_grad():
                reward[agent] -= args.diayn * discrim

            logs.update({f'diversity_loss_{i}': discrim.mean().item()})
            discrim_loss += discrim.mean()

        discrim_loss.backward()
        discrim_opt.step()

    if args.agent == 'gru':
        with torch.no_grad():
            # Reset hidden states on death or on environment reset
            for _agent, _done in done.items():
                if _agent != '__all__':
                    hidden_states[_agent | done['__all__']][_done].mul_(0)

    if args.agent != 'random' and args.train:
        trajectories.append(
            action=flatten_dict(actions),
            log_prob=flatten_dict(probs),
            value=flatten_dict(values),
            reward=flatten_dict(reward),
            done=flatten_dict({k: v for k, v in done.items() if k.startswith('agent_')}),
            entropy=flatten_dict(entropies)
        )

    if not args.train:
        hidden_states = {k: v.detach() for k, v in hidden_states.items()}

    ##########################
    # Advantage actor-critic #
    ##########################
    if args.agent != 'random' and args.train and i_step % args.update_steps == 0:
        with torch.no_grad():
            bootstrap_values = []
            for i, (agent, obs) in enumerate(observations.items()):
                # Get the model for the corresponding species
                species_idx = i * args.n_species // args.n_agents
                model = models[0 if args.share_backbone else species_idx]
                if agent_type == 'gru':
                    _, _bootstraps, _ = model(obs, hidden_states.get(agent))
                else:
                    _, _bootstraps = model(obs)

                if args.share_backbone:
                    # In this case we need to select the correct head of the network
                    _bootstraps = _bootstraps[:, species_idx]

                bootstrap_values.append(_bootstraps)

            bootstrap_values = torch.stack(bootstrap_values).view(args.n_envs*args.n_agents, 1).detach()

        value_loss, policy_loss = a2c.loss(
            bootstrap_values.detach(),
            trajectories.rewards,
            trajectories.values,
            trajectories.log_probs,
            torch.zeros_like(trajectories.dones) if args.mask_dones else trajectories.dones
        )

        entropy_loss = - trajectories.entropies.mean()

        for opt in optimizers:
            opt.zero_grad()
        loss = VALUE_LOSS_COEFF * value_loss + policy_loss + entropy_coeff * entropy_loss
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        for opt in optimizers:
            opt.step()

        trajectories.clear()
        hidden_states = {k: v.detach() for k, v in hidden_states.items()}

    ###########
    # Logging #
    ###########
    t = time() - t0
    num_episodes += done['__all__'].sum().item()
    num_steps += args.n_envs

    ewm_tracker(
        fps=args.n_envs/(time()-t_i)
    )

    if args.save_video:
        # If there is just one env save each episode to a different file
        # Otherwise save the whole video at the end
        if args.n_envs == 1:
            if done['__all__']:
                # Save video and make a new recorder
                recorder.close()
                recorder = VideoRecorder(env, path=PATH + f'/videos/{save_file}/{num_episodes}.mp4')

    logs.update({
        't': t,
        'steps': num_steps,
        'episodes': num_episodes,
        'env_done': done[f'__all__'].float().mean().item(),
    })
    for i in range(args.n_agents):
        logs.update({
            f'reward_{i}': reward[f'agent_{i}'].mean().item(),
            f'policy_entropy_{i}': entropies[f'agent_{i}'].mean().item(),
            f'done_{i}': done[f'agent_{i}'].float().mean().item(),
        })

        if args.env == 'snake':
            logs.update({
                f'boost_{i}': info[f'boost_{i}'][~done[f'agent_{i}']].float().mean().item(),
                f'food_{i}': info[f'food_{i}'][~done[f'agent_{i}']].mean().item(),
                f'edge_collisions_{i}': info[f'edge_collision_{i}'].float().mean().item(),
                f'snake_collisions_{i}': info[f'snake_collision_{i}'].float().mean().item(),
                f'size_{i}': info[f'size_{i}'][~done[f'agent_{i}']].mean().item(),
                # Return of an episode is equivalent to the size on death
                f'return_{i}': info[f'size_{i}'][done[f'agent_{i}']].mean().item(),
            })

        if args.env == 'laser':
            logs.update({
                f'hp_{i}': info[f'hp_{i}'].float().mean().item(),
                f'laser_{i}': info[f'action_7_{i}'].float().mean().item(),
                f'hit_rate_{i}': (info[f'action_7_{i}'] & reward[f'agent_{i}'].gt(EPS)).float().mean().item(),
                f'errors': env.errors.sum().item()
            })

    ewm_tracker(**logs)

    if i_step % PRINT_INTERVAL == 0:
        log_string = '[{:02d}:{:02d}:{:02d}]\t'.format(int((t // 3600)), int((t // 60) % 60), int(t % 60))
        log_string += 'Steps {:.2f}e6\t'.format(num_steps / 1e6)
        log_string += 'Entropy: {:.2e}\t'.format(ewm_tracker['policy_entropy_0'])
        log_string += 'Done: {:.2e}\t'.format(ewm_tracker['done_0'])
        log_string += 'Reward: {:.2e}\t'.format(ewm_tracker['reward_0'])

        if args.env == 'snake':
            log_string += 'Edge: {:.2e}\t'.format(ewm_tracker['edge_collisions_0'])
            log_string += 'Food: {:.2e}\t'.format(ewm_tracker['food_0'])
            log_string += 'Collision: {:.2e}\t'.format(ewm_tracker['snake_collisions_0'])
            log_string += 'Size: {:.3}\t'.format(ewm_tracker['size_0'])
            log_string += 'Boost: {:.2e}\t'.format(ewm_tracker['boost_0'])

        if args.env == 'laser':
            log_string += 'HP: {:.2e}\t'.format(ewm_tracker['hp_0'])
            log_string += 'Laser: {:.2e}\t'.format(ewm_tracker['laser_0'])

        log_string += 'FPS: {:.2e}\t'.format(ewm_tracker['fps'])

        print(log_string)

    if i_step % MODEL_INTERVAL == 0 and args.save_model:
        os.makedirs(os.path.split(f'{PATH}/models/{save_file}')[0], exist_ok=True)
        for i, model in enumerate(models):
            torch.save(model.state_dict(), f'{PATH}/models/{save_file}__species={i}.pt')

    if args.save_heatmap:
        head_heatmap += env.agents.view(args.n_envs, args.n_agents, args.size, args.size).sum(dim=0).clone()
        if i_step % HEATMAP_INTERVAL == 0:
            os.makedirs(f'{PATH}/heatmaps/{save_file}/', exist_ok=True)
            np.save(f'{PATH}/heatmaps/{save_file}/{num_steps}.npy', head_heatmap.cpu().numpy())
            # Reset heatmap
            head_heatmap = torch.zeros((args.n_agents, args.size, args.size),
                                       device=args.device, dtype=torch.float, requires_grad=False)

    if i_step % LOG_INTERVAL == 0 and args.save_logs:
        logger.write(logs)

    if num_steps > args.total_steps or num_episodes >= args.total_episodes:
        break

if args.save_video:
    recorder.close()
