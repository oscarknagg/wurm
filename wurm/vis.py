import matplotlib.pyplot as plt
import torch

from config import FOOD_CHANNEL, HEAD_CHANNEL, BODY_CHANNEL


def plot_envs(envs: torch.Tensor, env_idx: int = 0, mode: str = 'single'):
    """Plots a single environment from a batch of environments"""
    size = envs.shape[-1]

    if mode == 'single':
        img = (envs[env_idx, BODY_CHANNEL, :, :].numpy() > 0) * 0.5
        img += envs[env_idx, HEAD_CHANNEL, :, :].numpy() * 0.5
        img += envs[env_idx, FOOD_CHANNEL, :, :].numpy() * 1.5
        plt.imshow(img, vmin=0, vmax=1.5)
        plt.xlim((0, size-1))
        plt.ylim((0, size-1))
        plt.grid()
    elif mode == 'channels':
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, title in zip(range(3), ['Food','Head','Body']):
            axes[i].set_title(title)
            axes[i].imshow(envs[env_idx, i, :, :].numpy())
            axes[i].grid()
            axes[i].set_xlim((0, size-1))
            axes[i].set_ylim((0, size-1))
    else:
        raise Exception
