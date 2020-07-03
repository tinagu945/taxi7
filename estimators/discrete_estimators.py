import torch
from adversarial_learning.tau_list_dataset import TauListDataLoader
from policies.discrete_policy import DiscretePolicy


def q_estimator_discrete(pi_e, gamma, q, init_state_dist):
    """
    q-based estimator for policy value in discrete settings

    :param pi_e: discrete evaluation policy (should be of DiscretePolicy class)
    :param gamma: discount factor (0 < gamma <= 1)
    :param q: q network (assumed to be nn.Module, or some other class with
        identical __call__ semantics)
    :param init_state_dist: pytorch array of shape (num_s,), where num_s
        is the number of different states, contains probabilities of starting
        in each state
    :return: q-based policy value estimate
    """
    assert isinstance(pi_e, DiscretePolicy)
    num_s = init_state_dist.shape[0]
    s_all = torch.LongTensor(range(num_s))
    q_table = q(s_all)
    pi_table = pi_e.get_probability_table()
    # print(q_table.shape, pi_table.shape, init_state_dist.shape)
    mean_v = torch.einsum("sa,sa,s->", q_table, pi_table, init_state_dist)
    return float((1 - gamma) * mean_v)


def w_estimator_discrete(tau_list_data_loader, pi_e, pi_b, w):
    """
    w-based estimator for policy value in discrete settings

    :param tau_data_loader: data loader for trajectory list
        (should be of class TauListDataLoader)
    :param pi_e: discrete evaluation policy (should be of DiscretePolicy class)
    :param pi_b: discrete behavior policy (should be of DiscretePolicy class)
    :param w: w network (assumed to be nn.Module, or some other class with
        identical __call__ semantics)
    :return: w-based policy value estimate
    """
    assert isinstance(tau_list_data_loader, TauListDataLoader)
    weighted_reward_total = 0.0
    weighted_reward_norm = 0.0
    for s, a, _, r in tau_list_data_loader:
        pi_ratio = pi_e(s) / pi_b(s)
        eta_s_a = torch.gather(pi_ratio, dim=1, index=a.view(-1, 1)).view(-1)
        weighted_reward = float((r * w(s).view(-1) * eta_s_a).sum())
        weighted_reward_total += weighted_reward
        weighted_reward_norm += len(s)
    return weighted_reward_total / weighted_reward_norm

