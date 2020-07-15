import torch
import numpy as np
from environments.abstract_environment import AbstractEnvironment


def naive_reward_average_estimate(tau_list):
    """
    naively estimate policy value by averaging all observed rewards
    :param tau_list: list of trajectories, each of which is a
        tuple s, a, s_prime, r, where each of these is a pytorch tensor
    :return: naive reward average estimate
    """
    reward_tensor = torch.FloatTensor([r for _, _, _, r_tensor in tau_list
                                       for r in r_tensor])
    return reward_tensor.mean()


def on_policy_estimate(env, pi_e, gamma, num_tau, tau_len):
    """
    perform an on-policy estimate of pi_e by actually rolling out data using
    this evaluation policy

    :param env: environment to sample data from
        (should subclass AbstractEnvironment)
    :param pi_e: policy to evaluate (should be a policy from the
        policies module)
    :param gamma: discount factor (0.0 < gamma <= 1.0)
    :param tau_list: (optional) if this is provided, use this list of
        trajectories rather than generating new ones
    :return: on-policy estimate from tau_list discounted by gamma
    """
    assert isinstance(env, AbstractEnvironment)
    data = env.generate_roll_out(pi=pi_e, num_tau=num_tau,
                                 tau_len=tau_len, gamma=gamma)
    return float(data.r.mean())
