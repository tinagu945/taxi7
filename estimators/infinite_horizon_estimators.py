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
    v = lambda s_: (q(s_) * pi_e(s_)).sum(1)
    mean_v = init_state_sampler.compute_mean(v)
    return float((1 - gamma) * mean_v)


def w_estimator(tau_list_data_loader, pi_e, pi_b, w):
    """
    w-based estimator for policy value in discrete settings

    :param tau_data_loader: data loader for trajectory list
        (should be of class DataLoader)
    :param pi_e: evaluation policy (should be from policies module)
    :param pi_b: behavior policy (should be from policies module)
    :param w: w network (assumed to be nn.Module, or some other class with
        identical __call__ semantics)
    :return: w-based policy value estimate
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

