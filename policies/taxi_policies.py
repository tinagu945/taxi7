import numpy as np
from policies.discrete_policy import DiscretePolicy


def load_taxi_policy(path):
    pi_table = np.load(path)
    return DiscretePolicy(pi_table)