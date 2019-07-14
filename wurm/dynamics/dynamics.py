import torch
from torch.nn import functional as F

from wurm._filters import ORIENTATION_DELTAS


def move_pixels(pixels: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Takes in pixels and directions, returns updated heads

    Args:
        pixels:
        directions:
    """
    # Create head position deltas
    filters = ORIENTATION_DELTAS.to(dtype=pixels.dtype, device=pixels.device)
    head_deltas = F.conv2d(
        pixels,
        filters,
        padding=1
    )
    directions_onehot = F.one_hot(directions, 4).to(pixels.dtype)
    head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, directions_onehot]).unsqueeze(1)
    pixels += head_deltas
    return pixels