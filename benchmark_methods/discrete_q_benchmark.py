import torch
import numpy as np
from collections import defaultdict
from dataset.init_state_sampler import DiscreteInitStateSampler
from environments.taxi_environment import TaxiEnvironment

from policies.discrete_policy import DiscretePolicy
from policies.mixture_policies import MixtureDiscretePolicy
from policies.taxi_policies import load_taxi_policy
from estimators.infinite_horizon_estimators import q_estimator
from models.discrete_models import QTableModel
from utils.torch_utils import load_tensor_from_npy
from estimators.benchmark_estimators import on_policy_estimate


def fit_q_tabular(data, pi, gamma, max_num_q_iter=100000,
                  min_q_change=1e-7, verbose=False):
    """
    fit a q function in tabular setting, using observational data
    works by first estimating mean reward function and transition kernel,
    and then applying Bellman-style recurrence

    :param data: dataset we are using to fit (should be instance of
        TauListDataset)
    :param pi: policy we are fitting q function for
        (should be a function that can take as input a pytorch batch of
        states and actions, and return a corresponding batch of action
        probabilities, i.e. should be a policy class defined in policies)
    :param gamma: value to use for discounting (assume 0 < gamma <= 1)
    :param max_num_q_iter: maximum number of iterations of recurrence
    :param min_q_change: minimum change in Q table between iterations
        (in terms of L_2 distance between the arrays) at which point we break
    :return: fitted Q function
    """
    # check that policy class is valid (i.e. is discrete)
    assert isinstance(pi, DiscretePolicy)
    pi_table = pi.get_probability_table()
    num_s, num_a = pi_table.shape

    # iterate through trajectories to build required aggregate views of reward
    # and successor state distributions, and overall state frequency
    observed_reward_lists = defaultdict(list)
    observed_successor_freqs = defaultdict(lambda: torch.zeros(num_s))
    state_count = torch.zeros(num_s)
    s, a, ss, r = data.s, data.a, data.s_prime, data.r
    for i in range(len(s)):
        key = (int(s[i]), int(a[i]))
        observed_reward_lists[key].append(float(r[i]))
        observed_successor_freqs[key][int(ss[i])] += 1
        state_count[int(s[i])] += 1.0
    state_count[int(ss[-1])] += 1.0

    # now aggregate these observed distributions per (s, a) pair to
    # estimate the R and T tables
    r_table = torch.zeros(num_s, num_a)
    t_table = torch.zeros(num_s, num_a, num_s)
    for s in range(num_s):
        for a in range(num_a):
            r_list = observed_reward_lists[(s, a)]
            if r_list:
                r_table[s, a] = float(np.mean(r_list))
                s_freq = observed_successor_freqs[(s, a)]
                t_table[s, a] = s_freq / s_freq.sum()
            else:
                # this means that (s, a) pair never observed, so use defaults
                r_table[s, a] = 0.0
                t_table[s, a] = state_count / state_count.sum()

    # finally perform Q iteration
    q_table = torch.zeros((num_s, num_a))
    for i in range(max_num_q_iter):
        v_table = (q_table * pi_table).sum(1)
        q_table_next = r_table + torch.einsum("sat,t->sa", t_table,
                                              gamma * v_table)
        q_change = ((q_table_next - q_table) ** 2).mean() ** 0.5
        q_table = q_table_next
        if q_change < min_q_change:
            break
    if verbose:
        print("exited at iteration %d", i)
    return QTableModel(q_table)


def debug():
    env = TaxiEnvironment()
    gamma = 0.98
    alpha = 0.6
    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_other = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_other, pi_1_weight=alpha)

    init_state_dist_path = "taxi_data/init_state_dist.npy"
    init_state_dist = load_tensor_from_npy(init_state_dist_path).view(-1)
    init_state_sampler = DiscreteInitStateSampler(init_state_dist)

    tau_list = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=200000)
    q = fit_q_tabular(tau_list=tau_list, pi=pi_e, gamma=gamma)
    q_estimate = q_estimator(pi_e=pi_e, gamma=gamma, q=q,
                             init_state_sampler=init_state_sampler)
    on_policy_est = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
                                       num_tau=1000, tau_len=1000)
    squared_error = (q_estimate - on_policy_est) ** 2
    print("q-based estimate:", q_estimate)
    print("on-policy estimate:", on_policy_est)
    print("squared error:", squared_error)


if __name__ == "__main__":
    debug()


