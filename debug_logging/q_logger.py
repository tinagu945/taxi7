import random
import torch
from adversarial_learning.game_objectives import q_game_objective
from benchmark_methods.discrete_q_benchmark import fit_q_tabular
from estimators.benchmark_estimators import on_policy_estimate
from estimators.infinite_horizon_estimators import q_estimator


class AbstractQLogger(object):
    def __init__(self, env, pi_e, gamma):
        self.env = env
        self.pi_e = pi_e
        self.gamma = gamma

    def log(self, train_data_loader, val_data_loader, q, f, epoch):
        """
        perform single log for Q learning, free to do any debug_logging
        as required

        :param train_data_loader: data loader for training data
        :param val_data_loader: data loader for validation data
        :param q: current q function
        :param f: current adversary function
        :param epoch: current epoch number
        :return: None
        """
        raise NotImplementedError()

    def log_benchmark(self, train_data_loader, val_data_loader, q, epoch):
        """
        perform single log for Q learning benchmarks, where there is no f
        function and game objective, again free to do any debug_logging
        as required

        :param train_data_loader: data loader for training data
        :param val_data_loader: data loader for validation data
        :param q: current q function
        :param epoch: current epoch number
        :return: None
        """
        raise NotImplementedError()


class SimplePrintQLogger(AbstractQLogger):
    def __init__(self, env, pi_e, gamma, init_state_sampler,
                 oracle_tau_len=1000000):
        AbstractQLogger.__init__(self, env, pi_e, gamma)

        # train an oracle Q function that will be used for debug_logging error
        # in Q function
        tau_list_oracle = self.env.generate_roll_out(
            pi=self.pi_e, num_tau=1, tau_len=oracle_tau_len, gamma=gamma)
        self.q_oracle = fit_q_tabular(tau_list=tau_list_oracle, pi=self.pi_e,
                                      gamma=self.gamma)
        sample_idx = list(range(oracle_tau_len))
        random.shuffle(sample_idx)
        s, a, s_prime, r = tau_list_oracle[0]
        self.s_sample = s[sample_idx[:5]]
        self.a_sample = a[sample_idx[:5]]
        self.s_prime_sample = s_prime[sample_idx[:5]]
        self.r_sample = r[sample_idx[:5]]

        # calculate oracle estimate of policy value that will be compared
        # against during validation
        self.policy_val_oracle = on_policy_estimate(
            env=self.env, pi_e=self.pi_e, gamma=self.gamma,
            num_tau=None, tau_len=None, tau_list=tau_list_oracle)
        self.init_state_sampler = init_state_sampler

    def log_benchmark(self, train_data_loader, val_data_loader, q, epoch):
        print("Validation results for benchmark epoch %d" % epoch)

        # print Q function on the fixed sample
        print("Q function sample values:")
        print(q(self.s_sample))

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            # calculate error in Q function
            q_err_total = 0.0
            q_err_norm = 0.0
            for s, a, s_prime, r in data_loader:
                q_pred = q(s)
                q_true = self.q_oracle(s)
                q_err_total += ((q_pred - q_true) ** 2).mean(1).sum()
                q_err_norm += len(s)
            q_rmse = float((q_err_total / q_err_norm) ** 0.5)
            print("%s Q function RMSE: %f" % (which, q_rmse))

        q_of_s_a_sample = torch.gather(q(self.s_sample), dim=1,
                                       index=self.a_sample.view(-1, 1)).view(-1)
        v_of_ss_sample = (self.pi_e(self.s_prime_sample)
                          * q(self.s_prime_sample)).sum(1)
        print("v(ss) sample:", v_of_ss_sample)
        print("q(s,a) sample:", q_of_s_a_sample)
        print("eq sample:", (self.r_sample + self.gamma * v_of_ss_sample
                             - q_of_s_a_sample))
        eq_total = 0.0
        eq_squared_total = 0.0
        eq_norm = 0.0
        for s, a, s_prime, r in train_data_loader:
            q_of_s_a = torch.gather(q(s), dim=1, index=a.view(-1, 1)).view(-1)
            v_of_ss = (self.pi_e(s_prime) * q(s_prime)).sum(1)
            eq = r + self.gamma * v_of_ss - q_of_s_a
            eq_total += float(eq.sum())
            eq_squared_total += float((eq ** 2).sum())
            eq_norm += len(s)
        mean_eq = eq_total / eq_norm
        mean_eq_squared = eq_squared_total / eq_norm
        print("mean eq:", mean_eq)
        print("uniform gmm norm:", (mean_eq ** 2) / mean_eq_squared)

        # estimate policy value
        policy_val_estimate = q_estimator(
            pi_e=self.pi_e, gamma=self.gamma, q=q,
            init_state_sampler=self.init_state_sampler)
        square_error = (policy_val_estimate - self.policy_val_oracle) ** 2
        print("Policy value estimate squared error:", square_error)
        print("")

    def log(self, train_data_loader, val_data_loader, q, f, epoch):
        print("Validation results for epoch %d" % epoch)

        # print Q function on the fixed sample
        print("Q function sample values:")
        print(q(self.s_sample))

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            # calculate mean game objective
            obj_total = 0.0
            obj_norm = 0.0
            for s, a, s_prime, r in data_loader:
                _, neg_obj = q_game_objective(q, f, s, a, s_prime, r,
                                              self.pi_e, self.gamma)
                obj_total += float(-neg_obj) * len(s)
                obj_norm += len(s)
            mean_obj = obj_total / obj_norm
            print("%s mean objective: %f" % (which, mean_obj))

            # calculate error in Q function
            q_err_total = 0.0
            q_err_norm = 0.0
            for s, a, s_prime, r in data_loader:
                q_pred = q(s)
                q_true = self.q_oracle(s)
                q_err_total += ((q_pred - q_true) ** 2).mean(1).sum()
                q_err_norm += len(s)
            q_rmse = float((q_err_total / q_err_norm) ** 0.5)
            print("%s Q function RMSE: %f" % (which, q_rmse))

        q_of_s_a_sample = torch.gather(q(self.s_sample), dim=1,
                                       index=self.a_sample.view(-1, 1)).view(-1)
        f_of_s_a_sample = torch.gather(f(self.s_sample), dim=1,
                                       index=self.a_sample.view(-1, 1)).view(-1)
        v_of_ss_sample = (self.pi_e(self.s_prime_sample)
                          * q(self.s_prime_sample)).sum(1)
        print("v(ss) sample:", v_of_ss_sample)
        print("q(s,a) sample:", q_of_s_a_sample)
        print("f(s,a) sample:", f_of_s_a_sample)
        print("eq sample:", (self.r_sample + self.gamma * v_of_ss_sample
                             - q_of_s_a_sample))
        eq_total = 0.0
        eq_squared_total = 0.0
        eq_norm = 0.0
        for s, a, s_prime, r in train_data_loader:
            q_of_s_a = torch.gather(q(s), dim=1, index=a.view(-1, 1)).view(-1)
            v_of_ss = (self.pi_e(s_prime) * q(s_prime)).sum(1)
            eq = r + self.gamma * v_of_ss - q_of_s_a
            eq_total += float(eq.sum())
            eq_squared_total += float((eq ** 2).sum())
            eq_norm += len(s)
        mean_eq = eq_total / eq_norm
        mean_eq_squared = eq_squared_total / eq_norm
        print("mean eq:", mean_eq)
        print("uniform gmm norm:", (mean_eq ** 2) / mean_eq_squared)

        # estimate policy value
        policy_val_estimate = q_estimator(
            pi_e=self.pi_e, gamma=self.gamma, q=q,
            init_state_sampler=self.init_state_sampler)
        square_error = (policy_val_estimate - self.policy_val_oracle) ** 2
        print("Policy value estimate squared error:", square_error)
        print("")

