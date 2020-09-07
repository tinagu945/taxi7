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


def w_estimator(tau_list_data_loader, pi_e, pi_b, w_oracle):
    """
    w_oracle-based estimator for policy value in discrete settings
    Eq(6) in Liu's paper?

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
        weighted_reward = float((r * w_oracle(s).view(-1) * eta_s_a).sum())
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
        b_prob = w_oracle(s).softmax(-1)[:, -1]
        w_true = (1-b_prob)/b_prob
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


def debug():
    import torch
    from dataset.init_state_sampler import DiscreteInitStateSampler
    from environments.taxi_environment import TaxiEnvironment
    from models.discrete_models import StateEmbeddingModel, QTableModel
    from models.w_adversary_wrapper import WAdversaryWrapper
    from policies.mixture_policies import MixtureDiscretePolicy
    from utils.torch_utils import load_tensor_from_npy
    from policies.taxi_policies import load_taxi_policy
    from estimators.benchmark_estimators import on_policy_estimate

    env = TaxiEnvironment(discrete_state=True)
    gamma = 0.98
    alpha = 0.6
    tau_len = 200000  # // 10000
    burn_in = 100000  # // 10000
    w_path = 'logs/2020-08-03T16:16:12.759808/best_w.pt'
    q_path = 'logs/2020-08-03T02:33:10.992962/best_q.pt'
    init_state_dist_path = "taxi_data/init_state_dist.npy"

    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_s = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_s, pi_1_weight=alpha)

    init_state_dist = load_tensor_from_npy(init_state_dist_path).view(-1)
    init_state_sampler = DiscreteInitStateSampler(init_state_dist)

    q_table = torch.zeros((env.num_s, env.num_a))
    q = QTableModel(q_table)
    q.load_state_dict(torch.load(q_path))
    w_oracle = StateEmbeddingModel(num_s=env.num_s, num_out=1)
    w_oracle.load_state_dict(torch.load(w_path))
    print('w_oracle and q loaded!')

    test_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                      burn_in=burn_in)
    test_data_loader = test_data.get_data_loader(1024)
    print('finished data generation.')

    double_est = double_estimator(
        test_data_loader, pi_e, pi_b, w_oracle, q, gamma, init_state_sampler)
    print('double est', double_est)
    q_est = q_estimator(pi_e, gamma, q, init_state_sampler)
    print('q_est', q_est)
    w_est = w_estimator(test_data_loader, pi_e, pi_b, w_oracle)
    print('w_est', w_est)

    policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
                                           num_tau=1, tau_len=1000000)
    print('on_policy_est', policy_val_oracle)


if __name__ == "__main__":
    debug()
