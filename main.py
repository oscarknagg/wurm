"""Entry point for training, transferring and visualising agents."""
import argparse

import torch

from wurm.callbacks.core import CallbackList
from wurm.callbacks.render import Render
from wurm.resume import LocalResume, S3Resume
from wurm.callbacks.model_checkpoint import ModelCheckpoint
from wurm.interaction import MultiSpeciesHandler
from wurm.rl import A2CLoss, A2CTrainer
from wurm.observations import FirstPersonCrop
from wurm import core
from wurm import arguments
from wurm import utils
from wurm.callbacks import loggers
from config import PATH, EPS, INPUT_CHANNELS


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
    args.dtype = torch.float
elif args.dtype == 'half':
    args.dtype = torch.half
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
if args.resume_mode == 'local':
    resume_data = LocalResume().resume(args, save_file)
elif args.resume_mode == 's3':
    resume_data = S3Resume().resume(args, save_file)
else:
    raise ValueError('Unrecognised resume-mode.')

args.agent_location = resume_data.model_paths

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
env = arguments.get_env(args)
env.observation_fn = observation_function
num_actions = env.num_actions


###################
# Create agent(s) #
###################
models = arguments.get_models(args, env.num_actions)

interaction_handler = MultiSpeciesHandler(models, args.n_species, args.n_agents, args.agent_type)
if args.train:
    a2c = A2CLoss(gamma=args.gamma, normalise_returns=args.norm_returns, dtype=args.dtype,
                  use_gae=args.gae_lambda is not None, gae_lambda=args.gae_lambda)
    a2c_trainer = A2CTrainer(
        models=models, update_steps=args.update_steps, lr=args.lr, a2c_loss=a2c, interaction_handler=interaction_handler,
        value_loss_coeff=args.value_loss_coeff, entropy_loss_coeff=args.entropy_loss_coeff, mask_dones=args.mask_dones,
        max_grad_norm=args.max_grad_norm)
else:
    a2c_trainer = None


callbacks = [
    loggers.LogEnricher(env, resume_data.current_steps, resume_data.current_episodes),
    loggers.PrintLogger(env=args.env, interval=args.print_interval) if args.print_interval is not None else None,
    Render(env, args.fps) if args.render else None,
    ModelCheckpoint(
        f'{PATH}/experiments/{args.save_folder}/models',
        f'repeat={args.repeat}__' + 'steps={steps:.2e}__species={i_species}.pt',
        models,
        interval=args.model_interval,
        s3_bucket=args.s3_bucket,
        s3_filepath=f'{args.save_folder}/models/repeat={args.repeat}__' + 'steps={steps:.2e}__species={i_species}.pt'
    ) if args.save_model else None,
    loggers.CSVLogger(
        filename=f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv',
        header_comment=utils.get_comment(args),
        interval=args.log_interval,
        append=resume_data.resume,
        s3_bucket=args.s3_bucket,
        s3_filename=f'{args.save_folder}/logs/{save_file}.csv',
        s3_interval=args.s3_interval
    ) if args.save_logs else None,
    loggers.VideoLogger(
        env,
        f'{PATH}/experiments/{args.save_folder}/videos/'
    ) if args.save_video else None,
    loggers.HeatMapLogger(
        env,
        save_folder=f'{PATH}/experiments/{args.save_folder}/heatmaps/',
        interval=args.heatmap_interval
    ) if args.save_heatmap else None,
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
    initial_steps=resume_data.current_steps,
    total_steps=args.total_steps,
    initial_episodes=resume_data.current_episodes,
    total_episodes=args.total_episodes
)
environment_run.run()
