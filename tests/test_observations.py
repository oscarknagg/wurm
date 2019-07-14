import unittest
from time import sleep
import matplotlib.pyplot as plt
import torch

from wurm import observations
from wurm import envs
from wurm.envs.laser_tag.map_generators import MapFromString
from wurm.envs.laser_tag import maps
from config import DEFAULT_DEVICE


torch.random.manual_seed(0)
render_envs = False


class TestFirstPersonCropping(unittest.TestCase):
    def test_on_asymmetric_env(self):
        obs_fn = observations.FirstPersonCrop(
            first_person_rotation=True,
            in_front=11,
            behind=2,
            side=6,
            padding_value=127
        )
        env = envs.LaserTag(num_envs=1, num_agents=4, height=9, width=16, map_generator=MapFromString(maps.small3, DEFAULT_DEVICE),
                            observation_fn=obs_fn, device=DEFAULT_DEVICE)

        if render_envs:
            env.render()
            sleep(3)

        agent_obs = obs_fn.observe(env)

        for agent, obs in agent_obs.items():
            if render_envs:
                obs_npy = obs.permute((2, 3, 1, 0))[:, :, :, 0].cpu().numpy()
                plt.imshow(obs_npy)
                plt.show()
