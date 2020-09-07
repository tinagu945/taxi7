import torch
from environments.taxi_environment import TaxiEnvironment
from policies.mixture_policies import MixtureDiscretePolicy
from policies.taxi_policies import load_taxi_policy
from all_compare import Density_Ratio_discounted
from benchmark_methods.discrete_q_benchmark import fit_q_tabular
from dataset.init_state_sampler import DiscreteInitStateSampler
from utils.torch_utils import load_tensor_from_npy
from models.discrete_models import StateEmbeddingModel, QTableModel
from models.w_adversary_wrapper import WAdversaryWrapper
from estimators.benchmark_estimators import *
from estimators.infinite_horizon_estimators import *


def debug():
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

    test_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                      burn_in=burn_in)
    test_data_loader = test_data.get_data_loader(1024)
    SASR = test_data.restore_strcture(discrete=True)
    print('finished data generation.')
    policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
                                           num_tau=1, tau_len=1000000)
    print('policy_val_oracle', policy_val_oracle)
    is_est = importance_sampling_estimator(SASR, pi_b, pi_e, gamma)
    is_ess = importance_sampling_estimator_stepwise(SASR, pi_b, pi_e, gamma)
    print('is_est', is_est, 'is_ess', is_ess)

    # Masa's w method, only for discrete
    den_discrete = Density_Ratio_discounted(env.n_state, gamma)
    _, w_table = train_density_ratio(SASR, pi_b, pi_e, den_discrete, gamma)
    w_table = w_table.reshape(-1)
    w = StateEmbeddingModel(num_s=env.num_s, num_out=1)
    w.set_weights(torch.from_numpy(w_table))
    print('[Masa] fitted w')

    # Masa's q method is the same as fit_q_tabular
    q = fit_q_tabular(test_data, pi_e, gamma)
    print('[Masa] fitted q')

    q_est = q_estimator(pi_e, gamma, q, init_state_sampler)
    drl_est = double_estimator(
        test_data_loader, pi_e, pi_b, w, q, gamma, init_state_sampler)
    w_est = w_estimator(test_data_loader, pi_e, pi_b, w)
    print('[Masa] q_est', q_est, '[Masa] w_est',
          w_est, '[Masa] drl_est', drl_est)

    q_table = torch.zeros((env.num_s, env.num_a))
    q = QTableModel(q_table)
    q.load_state_dict(torch.load(q_path))
    w_oracle = StateEmbeddingModel(num_s=env.num_s, num_out=1)
    w_oracle.load_state_dict(torch.load(w_path))
    print('[ours] w_oracle and q loaded!')

    double_est = double_estimator(
        test_data_loader, pi_e, pi_b, w_oracle, q, gamma, init_state_sampler)
    print('[ours] drl_est', double_est)
    q_est = q_estimator(pi_e, gamma, q, init_state_sampler)
    print('[ours] q_est', q_est)
    w_est = w_estimator(test_data_loader, pi_e, pi_b, w_oracle)
    print('[ours] w_est', w_est)


if __name__ == "__main__":
    debug()
