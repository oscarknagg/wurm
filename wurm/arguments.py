import argparse


def get_bool(input_string: str) -> bool:
    return input_string.lower()[0] == 't'


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--env', type=str)
    parser.add_argument('--n-envs', type=int)
    parser.add_argument('--n-agents', type=int)
    parser.add_argument('--n-species', type=int, default=1)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--warm-start', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dtype', type=str, default='float')
    parser.add_argument('--repeat', default=None, type=int, help='Repeat number')
    return parser


def add_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--diayn', default=0, type=float)
    parser.add_argument('--value-loss-coeff', default=0.5, type=float)
    parser.add_argument('--entropy-loss-coeff', default=0.01, type=float)
    parser.add_argument('--max-grad-norm', default=0.5, type=float)
    parser.add_argument('--coord-conv', default=True, type=get_bool)
    parser.add_argument('--mask-dones', default=True, type=get_bool, help='Removes deaths from training trajectories.')
    parser.add_argument('--train', default=True, type=get_bool)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--gae-lambda', default=None, type=float)
    parser.add_argument('--update-steps', default=5, type=int)
    parser.add_argument('--total-steps', default=float('inf'), type=float)
    parser.add_argument('--total-episodes', default=float('inf'), type=float)
    parser.add_argument('--norm-returns', default=False, type=get_bool)
    parser.add_argument('--share-backbone', default=False, type=get_bool)
    return parser


def add_model_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--agent-type', type=str, default='gru')
    parser.add_argument('--agent-location', type=str, nargs='+')
    return parser


def add_observation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--obs-h', type=int)
    parser.add_argument('--obs-w', type=int)
    parser.add_argument('--obs-rotate', type=get_bool)
    parser.add_argument('--obs-in-front', type=int)
    parser.add_argument('--obs-behind', type=int)
    parser.add_argument('--obs-side', type=int)
    return parser


def add_snake_env_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--boost', default=True, type=get_bool)
    parser.add_argument('--boost-cost', type=float, default=0.25)
    parser.add_argument('--food-on-death', type=float, default=0.33)
    parser.add_argument('--reward-on-death', type=float, default=-1)
    parser.add_argument('--food-mode', type=str, default='random_rate')
    parser.add_argument('--food-rate', type=float, default=3e-4)
    parser.add_argument('--respawn-mode', type=str, default='any')
    parser.add_argument('--colour-mode', type=str, default='random')
    return parser


def add_laser_tag_env_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--laser-tag-map', nargs='+', type=str, default='random')
    return parser


def add_render_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--render', default=False, type=lambda x: x.lower()[0] == 't')
    parser.add_argument('--render-window-size', default=256, type=int)
    parser.add_argument('--render-cols', default=1, type=int)
    parser.add_argument('--render-rows', default=1, type=int)
    parser.add_argument('--fps', default=12, type=int)
    return parser


def add_output_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--print-interval', default=1000, type=int)
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument('--model-interval', default=1000, type=int)
    parser.add_argument('--heatmap-interval', default=1, type=int)
    parser.add_argument('--save-folder', type=str, default=None)
    parser.add_argument('--save-location', type=str, default=None)
    parser.add_argument('--save-model', default=True, type=get_bool)
    parser.add_argument('--save-logs', default=True, type=get_bool)
    parser.add_argument('--save-video', default=False, type=get_bool)
    parser.add_argument('--save-heatmap', default=False, type=get_bool)
    return parser
