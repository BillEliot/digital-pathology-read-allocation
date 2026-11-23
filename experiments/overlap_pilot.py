"""Cyclic reader blocks after the common prefix."""

import numpy as np


def schedule(size, n, shared):
    common = np.arange(shared)
    remainder = n - shared
    return np.stack([np.r_[common, shared + (np.arange(remainder) + r * remainder) % (size - shared)] for r in range(3)])
