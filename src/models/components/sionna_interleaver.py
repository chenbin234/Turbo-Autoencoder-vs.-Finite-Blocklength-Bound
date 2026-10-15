"""Permutation generation for the 5G polar uplink channel interleaver."""

from math import isqrt
from numbers import Integral

import torch


def interleaver_polar(length: int) -> torch.Tensor:
    """Generate Sionna's triangular uplink channel interleaver.

    Args:
        length (int): Nonnegative length of the interleaver.

    Returns:
        torch.Tensor: Permutation of ``range(length)`` with shape
            ``[length]`` and dtype ``torch.int64``.

    Raises:
        TypeError: If ``length`` is not an integer or is a boolean.
        ValueError: If ``length`` is negative.

    """
    # Find the smallest T with T * (T + 1) / 2 >= length, exactly.
    triangle_size = (isqrt(8 * length + 1) - 1) // 2
    if triangle_size * (triangle_size + 1) // 2 < length:
        triangle_size += 1

    # Row r contains T-r entries, filled consecutively from row_starts[r].
    row_starts = [row * triangle_size - row * (row - 1) // 2 for row in range(triangle_size)]

    # Read the triangle column by column, skipping unused trailing cells.
    permutation = []
    for column in range(triangle_size):
        for row in range(triangle_size - column):
            index = row_starts[row] + column
            if index < length:
                permutation.append(index)

    return torch.tensor(permutation, dtype=torch.int64)


if __name__ == "__main__":
    # Test the interleaver_polar function with a few examples.
    length = 65
    perm = interleaver_polar(length)
    print(perm.shape)
    print(perm.dtype)
    print(f"Length: {length}, Permutation: {perm.tolist()}")
