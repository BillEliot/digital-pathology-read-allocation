"""Reader pairs and orders for three-reader replays."""

from itertools import combinations, permutations

import numpy as np


PAIRS = tuple(combinations(range(3), 2))
ORDERS = np.array(tuple(permutations(range(3))))
