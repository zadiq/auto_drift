"""
Implementation of different metrics
"""
import numpy as np


def resolve_rw(rw):
    if rw in [np.nan, -np.nan]:
        return -999
    return rw


def drift_cost(desired_state, state, sigma=.01, scale=1):
    """
    Computes the cost/reward of current state in respect to
    desired state.
    Based on: Autonomous Drifting using Simulation-Aided Reinforcement
              Learning. (Eq. 7)

    :param desired_state: A Vector of desired state
    :param state: A Vector of current state
    :param sigma: A tolerance hyper-parameter. the bigger it is the bigger the
                  tolerated error gap between desired and real state.
    :param scale: A positive scaling factor
    :return: cost
    """
    return (1 - np.exp(-np.sum((desired_state - state) ** 2) / (2 * (sigma**2)))) * scale


def drift_reward(desired_state, state, sigma=.01, scale=1):
    """
    Scaled between (-1 and 0) * scale
    """
    return resolve_rw(-1 * drift_cost(desired_state, state, sigma, scale))


def drift_reward_v2(desired_state, state, **_):
    return resolve_rw(-np.sum((desired_state - state) ** 2))


def get_drift_reward(version="v2"):
    func_map = {
        "v1": drift_reward,
        "v2": drift_reward_v2,
    }
    return func_map[version]
