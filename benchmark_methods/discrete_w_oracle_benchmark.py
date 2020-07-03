import torch


def calculate_tabular_w_oracle(env, pi_b, pi_e, gamma, num_s,
                               tau_len=1000000, burn_in=100000):
    """
    :param env: environment (should be AbstractEnvironment)
    :param pi_b: behavior policy (should be from policies module)
    :param pi_e: evaluation policy (should be from policies module)
    :param gamma: discount factor
    :param num_s: number of different states
    :param tau_len: length to trajectory to use for monte-carlo estimate
    :param burn_in: burn-in period for monte-carlo sampling
    :return:
    """
    tau_list_e = env.generate_roll_out(pi=pi_e, num_tau=1, tau_len=tau_len,
                                       gamma=gamma, burn_in=burn_in)
    s_e = tau_list_e[0][0]
    tau_list_b = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                       burn_in=burn_in)
    s_b = tau_list_b[0][0]

    w = torch.zeros(num_s)
    for i in range(num_s):
        freq_e = float((s_e == i).sum())
        freq_b = float((s_b == i).sum())
        w[i] = freq_e / freq_b if freq_b > 0 else 1.0
    return w
