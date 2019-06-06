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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from wurm.envs import SingleSnake, SimpleGridworld, MultiSnake
from wurm import agents
from wurm.utils import CSVLogger, ExponentialMovingAverageTracker
from wurm.rl import A2C, TrajectoryStore
from config import BODY_CHANNEL, HEAD_CHANNEL, FOOD_CHANNEL, PATH


LOG_INTERVAL = 1
MODEL_INTERVAL = 10
PRINT_INTERVAL = 100
MAX_GRAD_NORM = 0.5
FPS = 10


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--n-envs', type=int)
parser.add_argument('--n-agents', type=int)
parser.add_argument('--size', type=int)
parser.add_argument('--agent', type=str)
parser.add_argument('--observation', type=str)
parser.add_argument('--n-species', type=int, default=1)
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
parser.add_argument('--update-steps', default=20, type=int)
parser.add_argument('--entropy', default=0.0, type=float)
parser.add_argument('--entropy-min', default=None, type=float)
parser.add_argument('--total-steps', default=float('inf'), type=float)
parser.add_argument('--total-episodes', default=float('inf'), type=float)
parser.add_argument('--save-location', type=str, default=None)
parser.add_argument('--save-model', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--save-logs', default=True, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--save-video', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--norm-returns', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--boost-cost', type=float, default=0.25)
parser.add_argument('--food-on-death', type=float, default=0.33)
parser.add_argument('--reward-on-death', type=float, default=-1)
parser.add_argument('--food-mode', type=str, default='random_rate')
parser.add_argument('--food-rate', type=float, default=None)
parser.add_argument('--respawn-mode', type=str, default='any')
parser.add_argument('--dtype', type=str, default='float')
parser.add_argument('--flicker', type=int, default=None)
parser.add_argument('--r', default=None, type=int, help='Repeat number')

args = parser.parse_args()

excluded_args = ['train', 'device', 'verbose', 'save_location', 'save_model', 'save_logs', 'render',
                 'render_window_size', 'render_rows', 'render_cols', 'save_video', 'env', 'coord_conv',
                 'norm_returns', 'dtype', 'food_mode', 'respawn_mode', 'boost', 'warm_start',
                 'flicker', 'entropy_min'
                 ]
included_args = ['n_envs', 'n_agents', 'n_species', 'lr', 'gamma', 'update_steps', 'food_rate', 'boost_cost', 'entropy',
                 'food_on_death']


entropy_coeff = args.entropy

if args.r is None:
    excluded_args += ['r', ]
if args.total_steps == float('inf'):
    excluded_args += ['total_steps']
if args.total_episodes == float('inf'):
    excluded_args += ['total_episodes']
argsdict = {k: v for k, v in args.__dict__.items() if k not in excluded_args}
argstring = '__'.join([f'{k}={v}' for k, v in argsdict.items()])
print(argstring)

if args.observation == 'full':
    observation_size = args.size
elif args.observation.startswith('partial_'):
    observation_size = 2*int(args.observation.split('_')[1]) + 1
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


###################
# Configure Agent #
###################
# Reload/fresh model
if os.path.exists(args.agent) or os.path.exists(os.path.join(PATH, 'models', args.agent)):
    # Agent is to be loaded from a file
    agent_path = args.agent if os.path.exists(args.agent) else os.path.join(PATH, 'models', args.agent)
    agent_str = args.agent.split('/')[-1][:-3]
    agent_params = {kv.split('=')[0]: kv.split('=')[1] for kv in agent_str.split('__')}
    agent_type = agent_params['agent']
    observation_type = agent_params['observation']
    reload = True
    print('Loading agent from file. Agent params:')
    pprint(agent_params)
else:
    # Creating a new agent
    agent_type = args.agent
    observation_type = args.observation
    reload = False

in_channels = 3
if args.boost:
    num_actions = 8  # each direction with/without boost
else:
    num_actions = 4

# Create agent(s)
# if args.n_agents % args.n_species
models: List[nn.Module] = []
if agent_type == 'relational':
    for i in range(args.n_species):
        models.append(
            agents.RelationalAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=in_channels, conv_channels=32, num_relational=2,
                num_attention_heads=2, relational_dim=32, num_feedforward=1, feedforward_dim=64, residual=True
            ).to(device=args.device, dtype=dtype)
        )
elif agent_type == 'conv':
    for i in range(args.n_species):
        models.append(
            agents.ConvAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=args.device, dtype=dtype)
        )
elif agent_type == 'gru':
    for i in range(args.n_species):
        models.append(
            agents.GRUAgent(
                num_actions=num_actions, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=args.device, dtype=dtype)
        )
elif agent_type == 'random':
    for i in range(args.n_species):
        models.append(agents.RandomAgent(num_actions=num_actions, device=args.device))
else:
    raise ValueError('Unrecognised agent')

if reload:
    models[0].load_state_dict(torch.load(agent_path))

if args.train:
    for m in models:
        m.train()
else:
    torch.no_grad()
    for m in models:
        m.eval()

print(models[0])

if args.agent != 'random' and args.train:
    optimizers: List[optim.Adam] = []
    for i in range(args.n_species):
        optimizers.append(
            optim.Adam(models[i].parameters(), lr=args.lr, weight_decay=1e-5)
        )
eps = np.finfo(np.float32).eps.item()


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
                     reward_on_death=args.reward_on_death)
    print('Reward on death')
    print(env.reward_on_death)
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
    comment += 'Input args:\n'
    comment += pformat(args.__dict__)
    logger = CSVLogger(filename=f'{PATH}/logs/{save_file}.csv', header_comment=comment)
if args.save_video:
    os.makedirs(PATH + f'/videos/{save_file}', exist_ok=True)
    recorder = VideoRecorder(env, path=PATH + f'/videos/{save_file}/0.mp4')

a2c = A2C(gamma=args.gamma, normalise_returns=args.norm_returns, dtype=dtype)


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
            model = models[i // args.n_species]
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

    #############################
    # Interact with environment #
    #############################
    actions = {}
    values = {}
    probs = {}
    entropies = {}
    for i, (agent, obs) in enumerate(observations.items()):
        model = models[i // args.n_species]
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
            all_obs = torch.stack([v for k, v in observations.items()]).view(
                -1, in_channels, observation_size, observation_size).detach()

            if agent_type == 'gru':
                all_hiddens = torch.stack([v for k, v in hidden_states.items()]).view(
                    -1, 64)

                _, bootstrap_values, _ = model(all_obs, all_hiddens)
            else:
                _, bootstrap_values = model(all_obs)

            # bootstrap_values = []
            # for i, (agent, obs) in enumerate(observations.items()):
            #     model = models[i // args.n_species]
            #     if agent_type == 'gru':
            #         _, _bootstraps, _ = model(obs, hidden_states.get(agent))
            #     else:
            #         _, _bootstraps = model(obs)
            #
            #     bootstrap_values.append(_bootstraps)
            #
            # bootstrap_values = torch.stack(bootstrap_values).view(args.n_envs*args.n_agents, 1).detach()

        bootstrap_values = torch.zeros_like(bootstrap_values)
        # print('bootstrap_values', bootstrap_values.shape)
        value_loss, policy_loss = a2c.loss(
            bootstrap_values.detach(),
            trajectories.rewards,
            trajectories.values,
            trajectories.log_probs,
            trajectories.dones
        )

        entropy_loss = - trajectories.entropies.mean()

        # optimizer.zero_grad()
        for opt in optimizers:
            opt.zero_grad()
        loss = value_loss + policy_loss + entropy_coeff * entropy_loss
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        # optimizer.step()
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

    # currently_alive = ~torch.stack([v for k, v in done.items() if k.startswith('agent_')]).t()
    # instantaneous_reward = torch.stack([r for a, r in reward.items()])
    # living_reward_rate = instantaneous_reward[currently_alive.t()].mean().item()
    # living_reward_rate = 0 if np.isnan(living_reward_rate) else living_reward_rate
    # instantaneous_entropy = torch.stack([r for a, r in entropies.items()])
    # living_entropy = instantaneous_entropy[currently_alive.t()].mean().item()
    # living_entropy = 0 if np.isnan(living_entropy) else living_entropy
    # instantaneous_edge = torch.stack([v for k, v in info.items() if k.startswith('edge_collision_')]).float().mean().item()
    # ewm_tracker(
    #     reward_rate=living_reward_rate,
    #     entropy=living_entropy,
    #     edge_collisions=instantaneous_edge,
    # )
    #
    # instanteous_sizes = env.bodies.view(env.num_envs, env.num_snakes, -1).max(dim=-1)[0]
    # living_sizes = instanteous_sizes[currently_alive].mean().item()
    # living_sizes = 0 if np.isnan(living_sizes) else living_sizes

    logs = {
        't': t,
        'steps': num_steps,
        'episodes': num_episodes,
    }
    for i in range(args.n_agents):
        logs.update({
            f'reward_{i}': reward[f'agent_{i}'][~done[f'agent_{i}']].mean(),
            f'policy_entropy_{i}': entropies[f'agent_{i}'][~done[f'agent_{i}']].mean(),
            f'food_{i}': info[f'food_{i}'][~done[f'agent_{i}']].mean(),
            f'edge_collisions_{i}': info[f'edge_collision_{i}'].float().mean(),
            f'snake_collisions_{i}': info[f'snake_collision_{i}'].float().mean(),
            f'boost_{i}': info[f'boost_{i}'][~done[f'agent_{i}']].float().mean(),
            f'size_{i}': info[f'size_{i}'][~done[f'agent_{i}']].mean(),
            f'done_{i}': done[f'agent_{i}'].float().mean(),

            f'__reward_{i}': reward[f'agent_{i}'].mean(),
            f'__size_{i}': info[f'size_{i}'].mean(),
        })

    ewm_tracker(**logs)

    # logs = {
    #     't': t,
    #     'steps': num_steps,
    #     'episodes': num_episodes,
    #     'reward_rate': living_reward_rate,
    #     'policy_entropy': living_entropy,
    #     'edge_collisions': instantaneous_edge,
    # }
    # if args.env == 'snake':
    #     instantaneous_snake = torch.stack([v for k, v in info.items() if k.startswith('snake_collision_')]).float().mean().item()
    #     instanteous_sizes = env.bodies.view(env.num_envs, env.num_snakes, -1).max(dim=-1)[0]
    #     living_sizes = instanteous_sizes[currently_alive].mean().item()
    #     living_sizes = 0 if np.isnan(living_sizes) else living_sizes
    #     instantaneous_snake_collision = \
    #         torch.stack([v for k, v in info.items() if k.startswith('snake_collision_')]).float().mean().item()
    #     # instaneous_boosts = torch.stack([v for k, v in env.boost_this_step.items()])
    #     instaneous_boosts = env.boost_this_step.view(env.num_snakes, env.num_envs)
    #     living_boosts = instaneous_boosts[currently_alive.t()].float().mean().item()
    #     living_boosts = 0 if np.isnan(living_boosts) else living_boosts
    #     ewm_tracker(
    #         snake_collisions=instantaneous_snake,
    #         avg_size=living_sizes,
    #         boost_rate=living_boosts
    #     )
    #     logs.update({
    #         'avg_size': living_sizes,
    #         'snake_collisions': instantaneous_snake_collision,
    #         'boost_rate': living_boosts
    #     })

    # if args.agent != 'random' and args.train and i_step > args.update_steps:
    #     logs.update({
    #         'value_loss': value_loss,
    #         'policy_loss': policy_loss,
    #         'entropy_loss': entropy_loss,
    #     })

    if i_step % PRINT_INTERVAL == 0:
        # pprint(info)
        log_string = '[{:02d}:{:02d}:{:02d}]\t'.format(int((t // 3600)), int((t // 60) % 60), int(t % 60))
        log_string += 'Steps {:.2f}e6\t'.format(num_steps / 1e6)
        log_string += 'Reward: {:.2e}\t'.format(ewm_tracker['reward_0'])
        log_string += 'Entropy: {:.2e}\t'.format(ewm_tracker['policy_entropy_0'])
        log_string += 'Done: {:.2e}\t'.format(ewm_tracker['done_0'])
        log_string += 'Edge: {:.2e}\t'.format(ewm_tracker['edge_collisions_0'])

        if args.env == 'snake':
            log_string += 'Collision: {:.2e}\t'.format(ewm_tracker['snake_collisions_0'])
            log_string += 'Size: {:.3}\t'.format(ewm_tracker['size_0'])
            log_string += 'Boost: {:.2e}\t'.format(ewm_tracker['food_0'])
        # if args.env == 'snake':
        #     log_string += 'Size: {:.3}\t'.format(ewm_tracker['avg_size'])
        #     log_string += 'Collision: {:.2e}\t'.format(ewm_tracker['snake_collisions'])
        #     log_string += 'Boost: {:.2e}\t'.format(ewm_tracker['boost_rate'])

        log_string += 'FPS: {:.2e}\t'.format(ewm_tracker['fps'])
        # log_string += 'Coeff: {:.2e}'.format(entropy_coeff)

        print(log_string)

    if i_step % MODEL_INTERVAL:
        if args.save_model:
            os.makedirs(f'{PATH}/models/', exist_ok=True)
            for i, model in enumerate(models):
                torch.save(model.state_dict(), f'{PATH}/models/{save_file}__species={i}.pt')

    if i_step % LOG_INTERVAL == 0:
        if args.save_logs:
            logger.write(logs)

    if num_steps >= args.total_steps or num_episodes >= args.total_episodes:
        break

if args.save_video:
    recorder.close()
