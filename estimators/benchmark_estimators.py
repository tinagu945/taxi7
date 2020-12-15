import os
import torch
import numpy as np
from environments.abstract_environment import AbstractEnvironment


# def naive_reward_average_estimate(tau_list):
#     """
#     naively estimate policy value by averaging all observed rewards
#     :param tau_list: list of trajectories, each of which is a
#         tuple s, a, s_prime, r, where each of these is a pytorch tensor
#     :return: naive reward average estimate
#     """
#     reward_tensor = torch.FloatTensor([r for _, _, _, r_tensor in tau_list
#                                        for r in r_tensor])
#     return reward_tensor.mean()


def on_policy_estimate(env=None, pi_e=None, gamma=None, num_tau=None, tau_len=None, pi_e_data_discounted=None):
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
    # 2 choices: generate data on the fly or get existing one.
    if not pi_e_data_discounted:
        assert isinstance(env, AbstractEnvironment)
        pi_e_data_discounted = env.generate_roll_out(pi=pi_e, num_tau=num_tau,
                                                     tau_len=tau_len, gamma=gamma)
    r = pi_e_data_discounted.r
    return float(r.mean())


def importance_sampling_estimator(SASR, pi_b, pi_e, gamma, split_shape=[1, 1, 1, 1]):
    mean_est_reward = 0.0
    for sasr in SASR:
        log_trajectory_ratio = 0.0
        total_reward = 0.0
        discounted_t = 1.0
        self_normalizer = 0.0

        for seq in sasr:
            s, a, sprime, r = torch.split(seq, split_shape)
            log_trajectory_ratio += np.log(torch.squeeze(pi_e(s))[int(a)]) - \
                np.log(torch.squeeze(pi_b(s))[int(a)])
            total_reward += r * discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
        avr_reward = total_reward / self_normalizer
        mean_est_reward += avr_reward * np.exp(log_trajectory_ratio)
    mean_est_reward /= len(SASR)
    return mean_est_reward


def importance_sampling_estimator_stepwise(SASR, pi_b, pi_e, gamma, split_shape=[1, 1, 1, 1]):
    mean_est_reward = 0.0
    for sasr in SASR:
        step_log_pr = 0.0
        est_reward = 0.0
        discounted_t = 1.0
        self_normalizer = 0.0
        for seq in sasr:
            s, a, sprime, r = torch.split(seq, split_shape)
            step_log_pr += np.log(torch.squeeze(pi_e(s))[int(a)]) - \
                np.log(torch.squeeze(pi_b(s))[int(a)])
            est_reward += np.exp(step_log_pr)*r*discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
        est_reward /= self_normalizer
        mean_est_reward += est_reward
    mean_est_reward /= len(SASR)
    return mean_est_reward


def train_density_ratio(SASR, pi_b, pi_e, den_discrete, gamma):
    for sasr in SASR:
        discounted_t = 1.0
        initial_state = sasr[0][0]
        for s, a, sprime, r in sasr:
            discounted_t = gamma
            policy_ratio = pi_e(s)[0][a]/pi_b(s)[0][a]
            den_discrete.feed_data(
                s, sprime, initial_state, policy_ratio, discounted_t)
        # den_discrete.feed_data(-1, initial_state, initial_state, 1, discounted_t)

    x, w = den_discrete.density_ratio_estimate()
    return x, w
