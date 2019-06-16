"""Entry point for training, transferring and visualising agents."""
import numpy as np
from typing import List
from itertools import count
from collections import namedtuple
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

from wurm.envs import SingleSnake, SimpleGridworld, MultiSnake
from wurm import agents
from wurm.utils import CSVLogger, ExponentialMovingAverageTracker, print_alive_tensors
from wurm.rl import A2C, TrajectoryStore
from config import BODY_CHANNEL, HEAD_CHANNEL, FOOD_CHANNEL, PATH


VALUE_LOSS_COEFF = 0.5
LOG_INTERVAL = 1
MODEL_INTERVAL = 1000
PRINT_INTERVAL = 1000
HEATMAP_INTERVAL = 1000
MAX_GRAD_NORM = 0.5
FPS = 10


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--n-envs', type=int)
parser.add_argument('--n-agents', type=int)
parser.add_argument('--n-species', type=int, default=1)
parser.add_argument('--size', type=int)
parser.add_argument('--agent', type=str, nargs='+')
parser.add_argument('--obs', type=str)
parser.add_argument('--warm-start', default=0, type=int)
parser.add_argument('--boost', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--train', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--coord-conv', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--render', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--render-window-size', default=256, type=int)
parser.add_argument('--render-cols', default=1, type=int)
parser.add_argument('--render-rows', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--gae-lambda', default=None, type=float)
parser.add_argument('--update-steps', default=20, type=int)
parser.add_argument('--entropy', default=0.0, type=float)
parser.add_argument('--entropy-min', default=None, type=float)
parser.add_argument('--total-steps', default=float('inf'), type=float)
parser.add_argument('--total-episodes', default=float('inf'), type=float)
parser.add_argument('--save-location', type=str, default=None)
parser.add_argument('--save-model', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--save-logs', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--save-video', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--save-heatmap', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--norm-returns', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--boost-cost', type=float, default=0.25)
parser.add_argument('--food-on-death', type=float, default=0.33)
parser.add_argument('--food-on-death-min', type=float, default=None)
parser.add_argument('--reward-on-death', type=float, default=-1)
parser.add_argument('--food-mode', type=str, default='random_rate')
parser.add_argument('--food-rate', type=float, default=3e-4)
parser.add_argument('--food-rate-min', type=float, default=None)
parser.add_argument('--respawn-mode', type=str, default='any')
parser.add_argument('--dtype', type=str, default='float')
parser.add_argument('--flicker', type=int, default=None)
parser.add_argument('--colour-mode', type=str, default='random')
parser.add_argument('--r', default=None, type=int, help='Repeat number')

args = parser.parse_args()

excluded_args = ['train', 'device', 'verbose', 'save_location', 'save_model', 'save_logs', 'render',
                 'render_window_size', 'render_rows', 'render_cols', 'save_video', 'env', 'coord_conv',
                 'norm_returns', 'dtype', 'food_mode', 'respawn_mode', 'boost', 'warm_start',
                 'flicker', 'entropy_min', 'update_steps',
                 ]
included_args = ['n_envs', 'n_agents', 'n_species', 'size', 'lr', 'gamma', 'update_steps', 'entropy', 'agent', 'obs', 'r']

entropy_coeff = args.entropy

if args.r is None:
    excluded_args += ['r', ]
if args.total_steps == float('inf'):
    excluded_args += ['total_steps']
if args.total_episodes == float('inf'):
    excluded_args += ['total_episodes']
argsdict = {k: v for k, v in args.__dict__.items() if k in included_args}
argstring = '__'.join([f'{k}={v}' for k, v in argsdict.items()])
print(argstring)

if args.obs == 'full':
    observation_size = args.size
elif args.obs.startswith('partial_'):
    observation_size = 2*int(args.obs.split('_')[1]) + 1
else:
    raise RuntimeError

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

# In channels
in_channels = 3
if args.boost:
    num_actions = 8  # each direction with/without boost
else:
    num_actions = 4

###################
# Create agent(s) #
###################
# Quick hack to make it easier to input all of the species trained in one particular experiment
if len(args.agent) == 1:
    species_0_path = args.agent[0]+'__species=0.pt'
    species_0_relative_path = os.path.join(PATH, 'models', args.agent[0])+'__species=0.pt'
    if os.path.exists(species_0_path) or os.path.exists(species_0_relative_path):
        agent_path = args.agent[0] if species_0_path else os.path.join(PATH, 'models', args.agent)
        args.agent = [agent_path + f'__species={i}.pt' for i in range(args.n_species)]

models: List[nn.Module] = []
for i in range(args.n_species):
    # Check for existence of model file
    model_path = args.agent[i]
    model_relative_path = os.path.join(PATH, 'models', args.agent[i])

    # Get agent params
    if os.path.exists(model_path) or os.path.exists(model_relative_path):
        agent_str = args.agent[i].split('/')[-1][:-3]
        agent_params = {kv.split('=')[0]: kv.split('=')[1] for kv in agent_str.split('__')}
        agent_type = agent_params['agent']
        observation_type = agent_params['obs']
        reload = True
    else:
        agent_type = args.agent[i]
        observation_type = args.obs
        reload = False

    # Create model class
    if agent_type == 'relational':
        models.append(
            agents.RelationalAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                num_relational=2,
                num_attention_heads=2, relational_dim=32, num_feedforward=1, feedforward_dim=64, residual=True
            ).to(device=args.device, dtype=dtype)
        )
    elif agent_type == 'conv':
        models.append(
            agents.ConvAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=args.device, dtype=dtype)
        )
    elif agent_type == 'gru':
        models.append(
            agents.GRUAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=args.device, dtype=dtype)
        )
    elif agent_type == 'random':
        models.append(agents.RandomAgent(num_actions=num_actions, device=args.device))
    else:
        raise ValueError('Unrecognised agent type.')

    # Load state dict if reload
    if reload:
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

print(models[0])

if agent_type != 'random' and args.train:
    optimizers: List[optim.Adam] = []
    for i in range(args.n_species):
        optimizers.append(
            optim.Adam(models[i].parameters(), lr=args.lr, weight_decay=1e-5)
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
    env = MultiSnake(num_envs=args.n_envs, num_snakes=args.n_agents, food_on_death_prob=args.food_on_death,
                     size=args.size, device=args.device, render_args=render_args, boost=args.boost,
                     boost_cost_prob=args.boost_cost, dtype=dtype, food_rate=args.food_rate,
                     respawn_mode=args.respawn_mode, food_mode=args.food_mode, observation_mode=observation_type,
                     reward_on_death=args.reward_on_death, agent_colours=args.colour_mode)
else:
    raise ValueError('Unrecognised environment')


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
            model = models[i * args.n_species // args.n_agents]
            if agent_type == 'gru':
                probs_, value_, hidden_states[agent] = model(obs, hidden_states.get(agent))
            else:
                probs_, value_ = model(obs)

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
        model = models[i * args.n_species // args.n_agents]
        if agent_type == 'gru':
            probs_, value_, hidden_states[agent] = model(obs, hidden_states.get(agent))
        else:
            probs_, value_ = model(obs)

        action_distribution = Categorical(probs_)
        actions[agent] = action_distribution.sample().clone().long()
        values[agent] = value_
        entropies[agent] = action_distribution.entropy()
        probs[agent] = action_distribution.log_prob(actions[agent].clone())

    observations, reward, done, info = env.step(actions)

    env.reset(done['__all__'], return_observations=False)
    env.check_consistency()

    if args.agent == 'gru':
        with torch.no_grad():
            # hidden_states_[done['agent_0']].mul_(0)
            # reset hidden states
            for _agent, _done in done.items():
                if _agent != '__all__':
                    hidden_states[_agent][_done].mul_(0)

    if args.agent != 'random' and args.train:
        # Flatten (num_envs, num_agents) tensor to (num_envs*num_agents, )
        # and then append to trajectory store
        all_actions = torch.stack([v for k, v in actions.items()]).view(-1, 1)
        all_log_probs = torch.stack([v for k, v in probs.items()]).view(-1)
        all_values = torch.stack([v for k, v in values.items()]).view(-1, 1)
        all_entropies = torch.stack([v for k, v in entropies.items()]).view(-1, 1)
        all_rewards = torch.stack([v for k, v in reward.items()]).view(-1, 1)
        all_dones = torch.stack([v for k, v in done.items() if k.startswith('agent_')]).view(-1, 1)

        trajectories.append(
            action=all_actions,
            log_prob=all_log_probs,
            value=all_values,
            reward=all_rewards,
            done=all_dones,
            entropy=all_entropies
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
                model = models[i * args.n_species // args.n_agents]
                if agent_type == 'gru':
                    _, _bootstraps, _ = model(obs, hidden_states.get(agent))
                else:
                    _, _bootstraps = model(obs)

                bootstrap_values.append(_bootstraps)

            bootstrap_values = torch.stack(bootstrap_values).view(args.n_envs*args.n_agents, 1).detach()

        value_loss, policy_loss = a2c.loss(
            bootstrap_values.detach(),
            trajectories.rewards,
            trajectories.values,
            trajectories.log_probs,
            trajectories.dones
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

    logs = {
        't': t,
        'steps': num_steps,
        'episodes': num_episodes,
    }
    for i in range(args.n_agents):
        logs.update({
            f'reward_{i}': reward[f'agent_{i}'][~done[f'agent_{i}']].mean().item(),
            f'policy_entropy_{i}': entropies[f'agent_{i}'][~done[f'agent_{i}']].mean().item(),
            f'food_{i}': info[f'food_{i}'][~done[f'agent_{i}']].mean().item(),
            f'edge_collisions_{i}': info[f'edge_collision_{i}'].float().mean().item(),
            f'snake_collisions_{i}': info[f'snake_collision_{i}'].float().mean().item(),
            f'boost_{i}': info[f'boost_{i}'][~done[f'agent_{i}']].float().mean().item(),
            f'size_{i}': info[f'size_{i}'][~done[f'agent_{i}']].mean().item(),
            f'done_{i}': done[f'agent_{i}'].float().mean().item(),
            # Return of an episode is equivalent to the size on death
            f'return_{i}': info[f'size_{i}'][done[f'agent_{i}']].mean().item(),
        })

    ewm_tracker(**logs)

    if i_step % PRINT_INTERVAL == 0:
        log_string = '[{:02d}:{:02d}:{:02d}]\t'.format(int((t // 3600)), int((t // 60) % 60), int(t % 60))
        log_string += 'Steps {:.2f}e6\t'.format(num_steps / 1e6)
        log_string += 'Reward: {:.2e}\t'.format(ewm_tracker['reward_0'])
        log_string += 'Entropy: {:.2e}\t'.format(ewm_tracker['policy_entropy_0'])
        log_string += 'Done: {:.2e}\t'.format(ewm_tracker['done_0'])
        log_string += 'Edge: {:.2e}\t'.format(ewm_tracker['edge_collisions_0'])

        if args.env == 'snake':
            log_string += 'Collision: {:.2e}\t'.format(ewm_tracker['snake_collisions_0'])
            log_string += 'Size: {:.3}\t'.format(ewm_tracker['size_0'])
            log_string += 'Boost: {:.2e}\t'.format(ewm_tracker['boost_0'])

        log_string += 'FPS: {:.2e}\t'.format(ewm_tracker['fps'])

        print(log_string)

    if i_step % MODEL_INTERVAL == 0and args.save_model:
        os.makedirs(f'{PATH}/models/', exist_ok=True)
        for i, model in enumerate(models):
            torch.save(model.state_dict(), f'{PATH}/models/{save_file}__species={i}.pt')

    if args.save_heatmap:
        head_heatmap += env.heads.view(args.n_envs, args.n_agents, args.size, args.size).sum(dim=0).clone()
        if i_step % HEATMAP_INTERVAL == 0:
            os.makedirs(f'{PATH}/heatmaps/{save_file}/', exist_ok=True)
            np.save(f'{PATH}/heatmaps/{save_file}/{num_steps}.npy', head_heatmap.cpu().numpy())
            # Reset heatmap
            head_heatmap = torch.zeros((args.n_agents, args.size, args.size),
                                       device=args.device, dtype=torch.float, requires_grad=False)

    if i_step % LOG_INTERVAL == 0 and args.save_logs:
        logger.write(logs)

    if num_steps >= args.total_steps or num_episodes >= args.total_episodes:
        break

if args.save_video:
    recorder.close()
