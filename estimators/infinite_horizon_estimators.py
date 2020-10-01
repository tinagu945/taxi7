import torch
from torch.utils.data import DataLoader
from dataset.init_state_sampler import AbstractInitStateSampler


def q_estimator(pi_e, gamma, q, init_state_sampler):
    """
    q-based estimator for policy value in discrete settings

    :param pi_e: evaluation policy (should be from policies module)
    :param gamma: discount factor (0 < gamma <= 1)
    :param q: q network (assumed to be nn.Module, or some other class with
        identical __call__ semantics)
    :param init_state_dist: should be an object that implements
        AbstractInitStateSampler, allows computing mean over initial state
        distribution
    :return: q-based policy value estimate
    """
    assert isinstance(init_state_sampler, AbstractInitStateSampler)
    def v(s_): return (q(s_) * pi_e(s_)).sum(1)
    mean_v = init_state_sampler.compute_mean(v)
    return float((1 - gamma) * mean_v)


def w_estimator(tau_list_data_loader, pi_e, pi_b, w):
    """
    w_oracle-based estimator for policy value in discrete settings
    Eq(6) in Liu's paper?

    :param tau_data_loader: data loader for trajectory list
        (should be of class DataLoader)
    :param pi_e: evaluation policy (should be from policies module)
    :param pi_b: behavior policy (should be from policies module)
    :param w: w network (assumed to be nn.Module, or some other class with
        identical __call__ semantics)
    :return: w_oracle-based policy value estimate
    """
    assert isinstance(tau_list_data_loader, DataLoader)
    weighted_reward_total = 0.0
    weighted_reward_norm = 0.0
    for s, a, _, r in tau_list_data_loader:
        pi_ratio = pi_e(s) / pi_b(s)
        eta_s_a = torch.gather(pi_ratio, dim=1, index=a.view(-1, 1)).view(-1)
        weighted_reward = float((r * w(s).view(-1) * eta_s_a).sum())
        weighted_reward_total += weighted_reward
        weighted_reward_norm += len(s)
    return weighted_reward_total / weighted_reward_norm


def oracle_w_estimator(tau_list_data_loader, pi_e, pi_b, w_oracle):
    """
    w_oracle-based estimator for policy value in discrete settings

    :param tau_data_loader: data loader for trajectory list
        (should be of class DataLoader)
    :param pi_e: evaluation policy (should be from policies module)
    :param pi_b: behavior policy (should be from policies module)
    :param w_oracle: w_oracle network (assumed to be nn.Module, or some other class with
        identical __call__ semantics)
    :return: w_oracle-based policy value estimate
    """
    assert isinstance(tau_list_data_loader, DataLoader)
    weighted_reward_total = 0.0
    weighted_reward_norm = 0.0
    for s, a, _, r in tau_list_data_loader:
        pi_ratio = pi_e(s) / pi_b(s)
        eta_s_a = torch.gather(pi_ratio, dim=1, index=a.view(-1, 1)).view(-1)
        b_prob = w_oracle(s).softmax(-1)[:, 0]
        w_true = (1 - b_prob) / (b_prob + 1e-7)
        w_true = w_true / w_true.mean()
        # print(w_true.mean())
        weighted_reward = float((r * w_true.view(-1) * eta_s_a).sum())
        weighted_reward_total += weighted_reward
        weighted_reward_norm += len(s)
    return weighted_reward_total / weighted_reward_norm


def double_estimator(tau_list_data_loader, pi_e, pi_b, w_oracle, q, gamma, init_state_sampler):
    """
    Eq (6) from Masatoshi's paper

    :param tau_data_loader: data loader for trajectory list
        (should be of class DataLoader)
    :param pi_e: evaluation policy (should be from policies module)
    :param pi_b: behavior policy (should be from policies module)
    :param w_oracle: w_oracle network (assumed to be nn.Module, or some other class with
        identical __call__ semantics)
    :return: w_oracle-based policy value estimate
    """
    assert isinstance(tau_list_data_loader, DataLoader)
    first_term = q_estimator(pi_e, gamma, q, init_state_sampler)

    weighted_reward_total = 0.0
    weighted_reward_norm = 0.0
    for s, a, s_prime, r in tau_list_data_loader:
        pi_ratio = pi_e(s) / pi_b(s)
        q_of_s_a = torch.gather(q(s), dim=1, index=a.view(-1, 1)).view(-1)
        v_of_ss = (pi_e(s_prime) * q(s_prime)).sum(1).detach()
        eta_s_a = torch.gather(pi_ratio, dim=1, index=a.view(-1, 1)).view(-1)
        weighted_reward = float(
            ((r + gamma * v_of_ss - q_of_s_a) * w_oracle(s).view(-1) * eta_s_a).sum())
        weighted_reward_total += weighted_reward
        weighted_reward_norm += len(s)
    second_term = weighted_reward_total / weighted_reward_norm
    return first_term+second_term
