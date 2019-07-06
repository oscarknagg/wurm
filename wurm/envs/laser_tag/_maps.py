import torch
from typing import List


def parse_mapstring(map: List[str]) -> (torch.Tensor, torch.Tensor):
    # Get height and width
    height = len(map)
    width = (len(map[0]) + 1) // 2

    # Check consistent height and width
    # Convert to tensor
    pathing = torch.zeros((1, 1, height, width))
    respawn = torch.zeros((1, 1, height, width))
    for i, line in enumerate(map):
        # Remove padding spaces
        line = (line + ' ')[::2]

        if len(line) != width:
            raise ValueError('Map string has inconsistent shape')

        _pathing = torch.tensor([char == '*' for char in line])
        pathing[:, :, i, :] = _pathing
        _respawn = torch.tensor([char == 'P' for char in line])
        respawn[:, :, i, :] = _respawn

    return pathing, respawn


small2_pathing = [
    '* * * * * * * * *',
    '* P           P *',
    '*               *',
    '*     *   *     *',
    '*   * *   * *   *',
    '*     *   *     *',
    '*               *',
    '* P           P *',
    '* * * * * * * * *',
]
small2a_pathing = [
    '* * * * * * * * *',
    '*       P       *',
    '*               *',
    '*     *   *     *',
    '* P * *   * * P *',
    '*     *   *     *',
    '*               *',
    '*       P       *',
    '* * * * * * * * *',
]
small2c_pathing = [
    '* * * * * * * * *',
    '*     *         *',
    '*   P       P   *',
    '* *   * * *   * *',
    '*       *       *',
    '*   P       P   *',
    '*       *       *',
    '*       *       *',
    '* * * * * * * * *',
]
small2b_pathing = [
    '* * * * * * * * *',
    '* P   *       P *',
    '*               *',
    '* *   * * *   * *',
    '*       *       *',
    '*               *',
    '*       *       *',
    '* P     *     P *',
    '* * * * * * * * *',
]


small3_pathing = [
    '* * * * * * * * * * * * * * * *',
    '* P                 *   * P   *',
    '*   * *       *               *',
    '*   *     *               *   *',
    '*                 *           *',
    '*   *                         *',
    '*   P     *           *       *',
    '*     *                     P *',
    '* * * * * * * * * * * * * * * *',
]
small3a_pathing = [
    '* * * * * * * * * * * * * * * *',
    '*             P     *   *     *',
    '*   * *       *               *',
    '*   *     *               *   *',
    '* P               *         P *',
    '*   *                         *',
    '*         *           *       *',
    '*     *          P            *',
    '* * * * * * * * * * * * * * * *',
]
small3b_pathing = [
    '* * * * * * * * * * * * * * * *',
    '* P         *               P *',
    '*           *                 *',
    '*                             *',
    '* * *   * * *                 *',
    '*           * * * *   * * * * *',
    '*                             *',
    '* P         *               P *',
    '* * * * * * * * * * * * * * * *',
]
small3c_pathing = [
    '* * * * * * * * * * * * * * * *',
    '*           *                 *',
    '*           *                 *',
    '*           P                 *',
    '* * * P * * *                 *',
    '*           * * * * P * * * * *',
    '*           P                 *',
    '*           *                 *',
    '* * * * * * * * * * * * * * * *',
]

small4_pathing = [
    '* * * * * * * * * * * * * * * * * * * * * *',
    '* P                 *   * P               *',
    '*   * *       *                     P     *',
    '*   *     *               *               *',
    '*                 *                     * *',
    '*   *                               *     *',
    '*   P     *           *                   *',
    '*     *                                 P *',
    '*                               *         *',
    '*           P                             *',
    '*                                         *',
    '*             *             P             *',
    '* *                                       *',
    '* * * * * * * * * * * * * * * * * * * * * *',
]
small4a_pathing = [
    '* * * * * * * * * * * * * * * * * * * * * *',
    '* P                 *   * P               *',
    '*   * *       *                     P     *',
    '*   *     *               *               *',
    '*                 *                     * *',
    '*   *                               *     *',
    '*   P     *           *                   *',
    '*     *                                 P *',
    '*                               *         *',
    '*           P                             *',
    '*                                         *',
    '*             *             P             *',
    '* *                                       *',
    '* * * * * * * * * * * * * * * * * * * * * *',
]
small4b_pathing = [
    '* * * * * * * * * * * * * * * * * * * * * *',
    '* P       *         P                     *',
    '*                           *             *',
    '*         *                 *           P *',
    '* * *   * * * * * *   * * * * * * * *   * *',
    '*         * P                 *           *',
    '*         *   *           *   *           *',
    '*                                         *',
    '* *   * * *                   *           *',
    '*         *   *           * P *           *',
    '*         *                   * *   * * * *',
    '*         * * * * *   * * * * *         P *',
    '* P                                       *',
    '* * * * * * * * * * * * * * * * * * * * * *',
]
small4c_pathing = [
    '* * * * * * * * * * * * * * * * * * * * * *',
    '*         *                               *',
    '*                           *             *',
    '*       P *       P         * P           *',
    '* * *   * * * * * *   * * * * * * * *   * *',
    '*         *                   *           *',
    '*       P *   *           *   *           *',
    '*                                         *',
    '* *   * * *                   * P         *',
    '*         *   *           *   *           *',
    '*         *                   * *   * * * *',
    '*       P * * * * *   * * * * * P         *',
    '*                                         *',
    '* * * * * * * * * * * * * * * * * * * * * *',
]

pathing, respawn =parse_mapstring(small2_pathing)
print(pathing)
print(respawn)
