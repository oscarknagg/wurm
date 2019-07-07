import torch
from typing import Dict, Optional, Tuple
from gym.envs.classic_control import rendering
import torch.nn.functional as F
from collections import OrderedDict
from time import time, sleep
import warnings

from wurm._filters import ORIENTATION_FILTERS
from wurm.utils import rotate_image_batch, drop_duplicates, pad_to_square, unpad_from_square
from wurm.core import MultiagentVecEnv, check_multi_vec_env_actions, build_render_rgb, move_pixels
from .map_generators import LaserTagMapGenerator
from wurm.observations import ObservationFunction, RenderObservations
from config import DEFAULT_DEVICE, EPS


def get_coords(input_tensor: torch.Tensor) -> torch.Tensor:
    batch_size, _, x_dim, y_dim = input_tensor.size()
    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    ret = torch.cat([
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

    return ret


class LaserTag(MultiagentVecEnv):
    """Laser tag environment.

    This environment is meant to be a slightly extended version of the laser tag multiagent
    environment from Deepmind's paper: https://arxiv.org/pdf/1711.00832.pdf
    """
    no_op = 0
    rotate_right = 1
    rotate_left = 2
    move_forward = 3
    move_back = 4
    move_right = 5
    move_left = 6
    fire = 7
    move_forward_and_turn_right = 8
    move_forward_and_turn_left = 9

    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second': 12
    }

    def __init__(self,
                 num_envs: int,
                 num_agents: int,
                 height: int,
                 width: int,
                 map_generator: LaserTagMapGenerator,
                 strict: bool = False,
                 verbose: int = 0,
                 colour_mode: str = 'random',
                 observation_fn: ObservationFunction = RenderObservations(),
                 manual_setup: bool = False,
                 initial_hp: int = 2,
                 render_args: dict = None,
                 env_lifetime: int = 1000,
                 dtype: torch.dtype = torch.float,
                 device: str = DEFAULT_DEVICE):
        super(LaserTag, self).__init__(num_envs, num_agents, height, width, dtype, device)
        self.strict = strict
        self.verbose = verbose
        self.map_generator = map_generator
        self.colour_mode = colour_mode
        self.observation_fn = observation_fn
        self.initial_hp = initial_hp
        self.max_env_lifetime = env_lifetime
        self.num_actions = 10

        if colour_mode == 'random':
            self.agent_colours = self._get_n_colours(num_envs*num_agents)
        elif colour_mode == 'fixed':
            fixed_colours = torch.tensor([
                [192, 0, 0],  # Red
                [0, 0, 192],  # Blue
                [0, 192, 0],  # Green
            ], device=device, dtype=torch.short)
            self.agent_colours = fixed_colours[:self.num_agents].repeat((self.num_envs, 1))
        else:
            raise ValueError('colour_mode not recognised.')

        if render_args is None:
            self.render_args = {'num_rows': 1, 'num_cols': 1, 'size': 256}
        else:
            self.render_args = render_args

        # Environment tensors
        self.env_lifetimes = torch.zeros((num_envs), dtype=torch.long, device=self.device, requires_grad=False)
        self.lasers = torch.zeros((num_envs * num_agents, 1, height, width), dtype=self.dtype, device=self.device,
                                  requires_grad=False)

        self.has_fired = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=self.device)
        if not manual_setup:
            self.agents, self.orientations, self.dones, self.hp, self.pathing, self.respawns = self._create_envs(num_envs)
        else:
            self.agents = torch.zeros((num_envs*num_agents, 1, height, width), device=device, requires_grad=False)
            self.orientations = torch.zeros((num_envs * num_agents), dtype=torch.long, device=self.device, requires_grad=False)
            self.dones = torch.zeros((num_envs*num_agents), dtype=torch.uint8, device=device, requires_grad=False)
            self.hp = torch.ones((num_envs * num_agents), dtype=torch.long, device=device, requires_grad=False) * self.initial_hp
            self.pathing = torch.zeros((num_envs, 1, height, width), device=device, requires_grad=False, dtype=torch.uint8)
            self.respawns = torch.zeros((num_envs, 1, height, width), device=device, requires_grad=False, dtype=torch.uint8)

        # Environment outputs
        self.rewards = torch.zeros(self.num_envs * self.num_agents, dtype=torch.float, device=self.device)
        self.dones = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=self.device)

    def _log(self, msg: str):
        if self.verbose > 0:
            print(msg)

    def _other_agents(self) -> torch.Tensor:
        t0 = time()
        other_agents = self.agents.view(self.num_envs, self.num_agents, self.height, self.width) \
            .sum(dim=1, keepdim=True).repeat_interleave(self.num_agents, 0) \
            .sub(self.agents) \
            .gt(EPS)
        self._log(f'Other agents: {1000 * (time() - t0)}ms')
        return other_agents

    def step(self, actions: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict):
        check_multi_vec_env_actions(actions, self.num_envs, self.num_agents)
        info = {}
        actions = torch.stack([v for k, v in actions.items()]).t().flatten()

        # Reset stuff
        self.lasers = torch.zeros((self.num_envs * self.num_agents, 1, self.height, self.width), dtype=self.dtype,
                                  device=self.device,
                                  requires_grad=False)
        self.rewards = torch.zeros(self.num_envs * self.num_agents, dtype=torch.float, device=self.device)
        self.dones = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=self.device)

        # Movement
        has_moved = actions == self.move_forward
        has_moved |= actions == self.move_back
        has_moved |= actions == self.move_right
        has_moved |= actions == self.move_left
        has_moved |= actions == self.move_forward_and_turn_right
        has_moved |= actions == self.move_forward_and_turn_left
        if torch.any(has_moved):
            t0 = time()
            # Keep the original positions so we can reset if an agent does a move
            # that's disallowed by pathing.
            original_agents = self.agents.clone()

            # Default movement is in the orientation direction
            directions = self.orientations.clone()
            # Backwards is reverse of the orientation direction
            directions[actions == self.move_back] = directions[actions == self.move_back] + 2
            # Right is orientation +1
            directions[actions == self.move_right] = directions[actions == self.move_right] + 1
            # Left is orientation -1
            directions[actions == self.move_left] = directions[actions == self.move_left] + 3
            # Keep in range(4)
            directions.fmod_(4)
            self.agents[has_moved] = move_pixels(self.agents[has_moved], directions[has_moved])
            self._log(f'Movement: {1000 * (time() - t0)}ms')

            t0 = time()
            # Resolve agent-wall pathing
            overlap = self.pathing.repeat_interleave(self.num_agents, 0) & (self.agents.gt(EPS))
            reset_due_to_pathing = overlap.view(self.num_envs * self.num_agents, -1).any(dim=1)
            if torch.any(reset_due_to_pathing):
                self.agents[reset_due_to_pathing] = original_agents[reset_due_to_pathing]
            self._log(f'Wall-agent pathing: {1000 * (time() - t0)}ms')

            t0 = time()
            # Resolve agent-agent pathing
            # TODO: Make one agent in each collision actually do the move
            overlap = self.agents.gt(EPS) & self._other_agents()
            reset_due_to_pathing = overlap.view(self.num_envs * self.num_agents, -1).any(dim=1)
            if torch.any(reset_due_to_pathing):
                self.agents[reset_due_to_pathing] = original_agents[reset_due_to_pathing]
            self._log(f'Agent-agent pathing: {1000 * (time() - t0)}ms')

        t0 = time()
        # Update orientations
        self.orientations[(actions == self.rotate_right) | (actions == self.move_forward_and_turn_right)] += 1
        self.orientations[(actions == self.rotate_left) | (actions == self.move_forward_and_turn_left)] += 3
        self.orientations.fmod_(4)
        self._log(f'Orientations: {1000 * (time() - t0)}ms')

        self.has_fired = actions == self.fire
        if torch.any(self.has_fired):
            other = ~F.one_hot(torch.arange(self.num_agents, device=self.device).repeat(self.num_envs), self.num_agents).byte()
            other_agents = self._other_agents()

            t0 = time()
            # Handle lasers
            lasers = torch.ones((self.num_envs*self.num_agents, 1, self.height, self.width), dtype=torch.uint8,
                                device=self.device)

            orientation_preprocessing = [
                (0, 0, 0),
                (1, 270, 90),
                (2, 180, 180),
                (3, 90, 270),
            ]

            # The _do_lasers() method calculates laser positions in a vectorised way but only for a particular
            # orientation. Hence I rotate each agent to the required orientation, apply the algorithm and then
            # rotate back to original position
            agents = self.agents.clone()
            pathing = self.pathing.repeat_interleave(self.num_agents, 0) + other_agents.gt(EPS)
            # Pad to square if the h != w because we can only use rotate_image_batch on
            # square images
            agents = pad_to_square(agents)
            pathing = pad_to_square(pathing, padding_value=1)
            lasers = pad_to_square(lasers, padding_value=1)
            lasers[~self.has_fired] = 0

            for orientation, rotation, _ in orientation_preprocessing:
                _orientation = self.orientations == orientation
                if torch.any(_orientation):
                    pathing[_orientation] = rotate_image_batch(pathing[_orientation], degree=rotation)
                    agents[_orientation] = rotate_image_batch(agents[_orientation], degree=rotation)

            self._log(f'Rotation pre-processing: {1000 * (time() - t0)}ms')

            lasers[self.has_fired] = self._laser_trajectories(
                agents=agents[self.has_fired],
                pathing=pathing[self.has_fired]
            )

            for orientation, _, reverse_rotation in orientation_preprocessing:
                _orientation = self.orientations == orientation
                if torch.any(_orientation):
                    lasers[_orientation] = rotate_image_batch(lasers[_orientation], degree=reverse_rotation)

            lasers = unpad_from_square(lasers, original_h=self.height, original_w=self.width)

            # For rendering
            agent_blocking = self.agents \
                .view(self.num_envs, self.num_agents, self.height, self.width) \
                .sum(dim=1, keepdim=True) \
                .repeat_interleave(self.num_agents, 0).gt(EPS)
            self.lasers += (lasers & ~agent_blocking).float()

            # Check for hits (https://www.youtube.com/watch?v=RaMIIpc46gM) and update HP
            lasers_ = lasers \
                .view(self.num_envs, self.num_agents, self.height, self.width) \
                .repeat_interleave(self.num_agents, 0) \
                .float()
            pathing = torch.einsum('nahw,na->nhw', [lasers_, other.float()]).unsqueeze(1)
            hit = (self.agents * pathing).gt(EPS).view(self.num_envs*self.num_agents, -1).any(dim=-1)
            self.hp -= hit.long()
            self.hp.relu_()
            self.dones |= self.hp.eq(EPS)

            # Give rewards
            other_agents_hit = other_agents.gt(EPS) & lasers.gt(EPS)
            reward = other_agents_hit.view(self.num_envs*self.num_agents, -1).sum(dim=-1).float()
            self.rewards += reward

            # Kill
            self.dones |= self.hp.lt(EPS)
            self.agents[self.dones] = 0

        self.env_lifetimes += 1
        observations = self._observe()
        dones = {f'agent_{i}': d for i, d in
                 enumerate(self.dones.clone().view(self.num_envs, self.num_agents).t().unbind())}
        # # Environment is done if its past the maximum episode length
        dones['__all__'] = self.env_lifetimes.gt(self.max_env_lifetime)

        rewards = {f'agent_{i}': d for i, d in
                   enumerate(self.rewards.clone().view(self.num_envs, self.num_agents).t().unbind())}

        info.update({
            f'hp_{i}': d for i, d in
            enumerate(self.hp.clone().view(self.num_envs, self.num_agents).t().unbind())
        })
        for i_action in range(self.num_actions):
            action_i = actions == i_action
            info.update({
                f'action_{i_action}_{i_agent}': d for i_agent, d in
                enumerate(action_i.clone().view(self.num_envs, self.num_agents).t().unbind())
            })

        return observations, rewards, dones, info

    def _laser_trajectories(self, agents: torch.Tensor, pathing: torch.Tensor) -> torch.Tensor:
        """Calculates laser trajectories.

        Args:
            agents: Tensor of shape (num_agents, ...
            pathing: Tensor of shape (num_agents, ...
        """
        t0 = time()
        n = agents.size(0)
        coords = get_coords(agents)
        lasers = torch.ones((n, 1, self.height, self.width), dtype=torch.uint8,
                            device=self.device)

        lasers = pad_to_square(lasers, padding_value=1)

        xy = agents * coords
        x, y = xy.view(n, 2, -1).sum(dim=2).reshape(n, -1).unbind(dim=1)

        # Handle orientation 0
        lasers &= coords[:, 0:1] >= x[:, None, None, None].float()
        lasers &= coords[:, 1:2] == y[:, None, None, None].float()
        in_front_mask = (coords[:, 0:1] >= x[:, None, None, None].float())
        in_front_mask &= (coords[:, 1:] >= y[:, None, None, None].float())
        trimmed_pathing = pathing & in_front_mask

        # Can I do this without the doing a cumsum twice in each direction?
        # The difficulty is that I need the trajectory to be not continue through single block objects
        # yet also penetrate 1 block in to them for hit calculations downstream
        block = trimmed_pathing.cumsum(dim=2).cumsum(dim=3).cumsum(dim=2).cumsum(dim=3).gt(1+EPS)

        lasers &= ~block

        self._log(f'Lasers: {1000 * (time() - t0)}ms')

        return lasers

    def reset(self, done: torch.Tensor = None, return_observations: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        # Reset envs that contain no snakes
        if done is None:
            done = self.dones.view(self.num_envs, self.num_agents).all(dim=1)

        if self.colour_mode == 'random':
            # Change agent colours each death
            new_colours = self._get_n_colours(self.dones.sum().item())
            self.agent_colours[self.dones] = new_colours

        # Reset any environment which has passed the maximum lifetime. This will also regenerate the pathing
        # map if the pathing generator is randomised
        if torch.any(done):
            num_done = done.sum().item()
            agent_dones = done.repeat_interleave(self.num_agents)
            new_agents, new_orientations, new_dones, new_hp, new_pathing, new_respawns = self._create_envs(num_done)
            self.agents[agent_dones] = new_agents
            self.orientations[agent_dones] = new_orientations
            self.dones[agent_dones] = new_dones
            self.hp[agent_dones] = new_hp
            # If we are using a fixed MapGenerator these lines won't do anything
            self.pathing[done] = new_pathing
            self.respawns[done] = new_respawns

            self.env_lifetimes[done] = 0

        # Single agent resets
        if torch.any(self.dones):
            first_done_per_env = (self.dones.view(self.num_envs, self.num_agents).cumsum(
                dim=1) == 1).flatten() & self.dones

            pathing = self.pathing.clone()
            pathing |= self.agents\
                .view(self.num_envs, self.num_agents, self.height, self.width)\
                .gt(EPS)\
                .any(dim=1, keepdim=True)
            pathing = pathing.repeat_interleave(self.num_agents, 0)
            pathing = pathing[first_done_per_env]

            available_respawn_locations = self.respawns.repeat_interleave(self.num_agents, 0)
            available_respawn_locations = available_respawn_locations[first_done_per_env]

            new_agents, new_orientations, sucessfully_respawned = self._respawn(pathing, available_respawn_locations, False)

            self.agents[first_done_per_env] = new_agents
            self.orientations[first_done_per_env] = new_orientations
            self.hp[first_done_per_env] = self.initial_hp
            self.dones[first_done_per_env] = sucessfully_respawned

        if return_observations:
            return self._observe()

    def check_consistency(self):
        self.errors = torch.zeros_like(self.errors)

        # Overlapping agents
        overlapping_agents = self.agents.view(self.num_envs, self.num_agents, self.height, self.width) \
            .sum(dim=1, keepdim=True) \
            .view(self.num_envs, -1) \
            .max(dim=-1)[0] \
            .gt(1)
        self.errors |= overlapping_agents
        if torch.any(overlapping_agents):
            msg = 'Agent-agent overlap in {} envs.'.format(overlapping_agents.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # Pathing overlap
        agent_pathing_overlap = self.pathing.repeat_interleave(self.num_agents, 0) & self.agents.gt(EPS)
        if torch.any(agent_pathing_overlap):
            bad_envs = agent_pathing_overlap.view(self.num_envs * self.num_agents, -1).any(dim=1)
            msg = 'Agent-wall overlap in {} envs.'.format(bad_envs.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # Can only get reward when firing
        not_fired_and_reward = (~self.has_fired) & self.rewards.gt(EPS)
        if torch.any(not_fired_and_reward):
            raise RuntimeError('Agent has got reward without firing.')

        # Max reward per step is 1
        impossible_rewards = self.rewards.gt(1)
        if torch.any(impossible_rewards):
            msg = 'Agent has received impossibly high reward in {} envs'.format(impossible_rewards.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # Every alive agent must actually exist
        alive = ~self.dones
        agent_exists = self.agents.view(self.num_envs*self.num_agents, -1).sum(dim=1).gt(EPS)
        alive_and_doesnt_exist = alive & (~agent_exists)
        if torch.any(alive_and_doesnt_exist):
            msg = 'Agent is alive but doesn\'t exist in {} envs.'.format(alive_and_doesnt_exist.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # # Agents can only be dead if a laser has been fired by another agent this step
        # laser_fired_in_env = self.has_fired\
        #     .view(self.num_envs, self.num_agents)\
        #     .any(dim=1)\
        #     .repeat_interleave(self.num_agents, 0)

        # laser_pathing_overlap = self.pathing.repeat_interleave(self.num_agents, 0) & self.lasers.gt(EPS)
        # laser_pathing_overlap[:, :, :1, :] = 0
        # laser_pathing_overlap[:, :, -1:, :] = 0
        # laser_pathing_overlap[:, :, :, :1] = 0
        # laser_pathing_overlap[:, :, :, -1:] = 0
        # if torch.any(laser_pathing_overlap):
        #     bad_envs = laser_pathing_overlap.view(self.num_envs*self.num_agents, -1).any(dim=1)
        #     print(self.pathing.repeat_interleave(self.num_agents, 0)[bad_envs][0].long())
        #     print(self.lasers[bad_envs][0].long())
        #     render_env = torch.arange(self.num_envs*self.num_agents)[bad_envs][0].item() // self.num_agents
        #     self.render(env=render_env)
        #     sleep(5)
        #     raise RuntimeError('Laser-wall overlap')

        # Reset bad envs. I choose not to reset the env lifetime so we can maintain the nice
        # thing of having all of the envs terminate synchronously.
        if torch.any(self.errors):
            num_done = self.errors.sum().item()
            agent_dones = self.errors.repeat_interleave(self.num_agents)
            new_agents, new_orientations, new_dones, new_hp, new_pathing, new_respawns = self._create_envs(num_done)
            self.agents[agent_dones] = new_agents
            self.orientations[agent_dones] = new_orientations
            self.dones[agent_dones] = new_dones
            self.hp[agent_dones] = new_hp
            # If we are using a fixed MapGenerator this line won't do anything
            self.pathing[self.errors] = new_pathing
            self.respawns[self.errors] = new_respawns

    def _respawn(self, pathing: torch.Tensor, respawns: torch.Tensor, exception_on_failure: bool):
        """Respawns an agent by picking randomly from allowed locations.

        Args:
            pathing: Pathing maps for the a batch of environments. This should be a Tensor of
                shape (n, 1, h, w) and dtype uint8.
            respawns: Maps of allowed respawn positions for a batch of environments. This should be a Tensor of
                shape (n, 1, h, w) and dtype uint8
            exception_on_failure:
        """
        n = pathing.shape[0]

        respawns &= ~pathing

        respawn_indices = drop_duplicates(torch.nonzero(respawns), 0)

        new_agents = torch.sparse_coo_tensor(
            respawn_indices.t(), torch.ones(len(respawn_indices)), respawns.shape,
            device=self.device, dtype=self.dtype
        ).to_dense()

        # Change orientation randomly
        new_orientations = torch.randint(0, 4, size=(n, ), device=self.device)

        dones = torch.zeros(n, dtype=torch.uint8, device=self.device)

        return new_agents, new_orientations, dones

    def _create_envs(self, num_envs: int) -> Tuple[torch.Tensor, ...]:
        """Gets pathing from pathing generator and repeatedly respawns agents until torch.all(dones == 0).

        Args:
            num_envs: Number of envs to create
        """
        dones = torch.ones(num_envs*self.num_agents, dtype=torch.uint8, device=self.device)

        map = self.map_generator.generate(num_envs)
        respawns = map.respawn
        pathing = map.pathing

        agents = torch.zeros((num_envs * self.num_agents, 1, self.height, self.width), dtype=self.dtype,
                             device=self.device, requires_grad=False)
        orientations = torch.zeros((num_envs * self.num_agents), dtype=torch.long, device=self.device,
                                   requires_grad=False)
        hp = torch.ones(
            (num_envs * self.num_agents), dtype=torch.long, device=self.device, requires_grad=False) * self.initial_hp

        while torch.any(dones):
            first_done_per_env = (dones.view(num_envs, self.num_agents).cumsum(
                dim=1) == 1).flatten() & dones

            _pathing = pathing.clone().repeat_interleave(self.num_agents, 0)
            _pathing |= agents \
                .view(num_envs, self.num_agents, self.height, self.width) \
                .gt(EPS) \
                .any(dim=1, keepdim=True)\
                .repeat_interleave(self.num_agents, 0)
            _pathing = _pathing[first_done_per_env]

            available_respawn_locations = respawns.repeat_interleave(self.num_agents, 0)
            available_respawn_locations = available_respawn_locations[first_done_per_env]

            new_agents, new_orientations, sucessfully_respawned = self._respawn(_pathing, available_respawn_locations,
                                                                                False)

            agents[first_done_per_env] += new_agents
            orientations[first_done_per_env] = new_orientations
            hp[first_done_per_env] = self.initial_hp
            dones[first_done_per_env] = sucessfully_respawned

        return agents, orientations, dones, hp, pathing, respawns

    def _get_env_images(self) -> torch.Tensor:
        t0 = time()
        # Black background
        img = torch.zeros((self.num_envs, 3, self.height, self.width), device=self.device, dtype=torch.short)

        # Add slight highlight for orientation
        locations = (self.agents > EPS).float()
        filters = ORIENTATION_FILTERS.to(dtype=self.dtype, device=self.device)
        orientation_highlights = F.conv2d(
            locations,
            filters,
            padding=1
        ) * 31
        directions_onehot = F.one_hot(self.orientations, 4).to(self.dtype)
        orientation_highlights = torch.einsum('bchw,bc->bhw', [orientation_highlights, directions_onehot]) \
            .view(self.num_envs, self.num_agents, 1, self.height, self.width) \
            .sum(dim=1) \
            .short()
        img += orientation_highlights.expand_as(img)

        # Add lasers
        per_env_lasers = self.lasers.reshape(self.num_envs, self.num_agents, self.height, self.width).sum(dim=1).gt(EPS)
        # Convert to NHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))
        laser_colour = torch.tensor([127, 127, 31], device=self.device, dtype=torch.short)
        img[per_env_lasers] = laser_colour
        # Convert back to NCHW axes
        img = img.permute((0, 3, 1, 2))

        # Add colours for agents
        body_colours = torch.einsum('nhw,nc->nchw', [locations.squeeze(), self.agent_colours.float()]) \
            .view(self.num_envs, self.num_agents, 3, self.height, self.width) \
            .sum(dim=1) \
            .short()
        img += body_colours

        # Walls are grey
        img[self.pathing.expand_as(img)] = 127
        self._log(f'Rendering: {1000 * (time() - t0)}ms')

        return img

    def _observe(self) -> Dict[str, torch.Tensor]:
        t0 = time()
        observations = self.observation_fn.observe(self)
        self._log(f'Observations: {1000 * (time() - t0)}ms')
        return observations

    def _get_n_colours(self, n: int) -> torch.Tensor:
        colours = torch.rand((n, 3), device=self.device)
        colours /= colours.norm(2, dim=1, keepdim=True)
        colours *= 192
        colours = colours.short()
        return colours

    @property
    def x(self):
        return self.agents.view(self.num_envs*self.num_agents, -1).argmax(dim=1) // self.width

    @property
    def y(self):
        return self.agents.view(self.num_envs*self.num_agents, -1).argmax(dim=1) % self.width
