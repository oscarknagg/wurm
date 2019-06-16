"""Takes a folder full of agent files and runs many random """
import argparse
import os
import numpy as np

from config import PATH


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--n-envs', type=int)
parser.add_argument('--n-agents', type=int)
parser.add_argument('--size', type=int)
parser.add_argument('--agents-folder', type=str)
parser.add_argument('--n-rounds', type=int)
parser.add_argument('--obs', type=str)
parser.add_argument('--with-replacement', default=False, type=lambda x: x.lower()[0] == 't')
parser.add_argument('--total-steps', default=float('inf'), type=float)
parser.add_argument('--warm-start', default=0, type=int)
parser.add_argument('--boost', default=True, type=lambda x: x.lower()[0] == 't')
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
parser.add_argument('--colour-mode', type=str, default='random')
parser.add_argument('--dtype', type=str, default='float')
args = parser.parse_args()


agent_paths = []
for root, _, agents in os.walk(os.path.join(PATH, args.agents_folder)):
    for a in agents:
        agent_paths.append(os.path.join(root, a))


random_matchups = []
for i in range(args.n_rounds):
    players = np.random.choice(agent_paths, size=args.n_agents, replace=args.with_replacement)
    random_matchups.append(players)


included_args = ['env', 'n_envs', 'n_agents', 'size', 'obs', 'total_steps', 'boost', 'device', 'boost_cost',
                 'food_on_death', 'reward_on_death', 'food_mode', 'food_rate', 'respawn_mode', 'colour_mode',
                 'dtype', 'save_location']
save_args = ['n-envs', 'n-agents']
for i, players in enumerate(random_matchups):
    eval_args = {k: v for k, v in args.__dict__.items() if k in included_args}
    eval_args['n_species'] = eval_args['n_agents']
    eval_args = {k.replace('_', '-'): v for k, v in eval_args.items()}

    eval_command = 'python -m experiments.multiagent \\\n'
    for arg, val in eval_args.items():
        eval_command += f'\t--{arg} {val} \\\n'

    eval_command += f'\t--agent \\\n'
    for p in players:
        eval_command += f'\t\t{p} \\\n'

    # Build save location
    save_location = '__'.join([f'{k}={v}' for k, v in eval_args.items() if k in save_args])
    save_location += f'__r={i}'
    save_location = save_location.replace('-', '_')
    eval_command += f'\t--save-location {save_location}'

    print(eval_command)
    os.system(eval_command)
