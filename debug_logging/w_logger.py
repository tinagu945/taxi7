import random

from adversarial_learning.game_objectives import w_game_objective
from dataset.init_state_sampler import AbstractInitStateSampler
from benchmark_methods.discrete_w_oracle_benchmark import \
    calculate_tabular_w_oracle
from estimators.benchmark_estimators import on_policy_estimate
from estimators.infinite_horizon_estimators import w_estimator


class AbstractWLogger(object):
    def __init__(self, env, pi_e, pi_b, gamma):
        self.env = env
        self.pi_e = pi_e
        self.pi_b = pi_b
        self.gamma = gamma

    def log(self, train_data_loader, val_data_loader, w, f,
            init_state_sampler, epoch):
        """
        perform single log for Q learning, free to do any debug_logging as
        required

        :param val_data_loader: data loader for validation data
        :param w: current w function
        :param f: current adversary function
        :param init_state_sampler: should be a subclass of
            AbstractInitStateSampler
        :param epoch: current epoch number
        :return: None
        """
        raise NotImplementedError()


class SimplePrintWLogger(AbstractWLogger):
    def __init__(self, env, pi_e, pi_b, gamma, oracle_tau_len=1000000):
        AbstractWLogger.__init__(self, env, pi_e, pi_b, gamma)

        # estimate oracle W vector
        tau_list_oracle = self.env.generate_roll_out(
            pi=self.pi_e, num_tau=1, tau_len=oracle_tau_len, gamma=gamma)
        sample_idx = list(range(oracle_tau_len))
        random.shuffle(sample_idx)
        self.s_sample = tau_list_oracle[0][0][sample_idx[:5]]

        # calculate oracle estimate of policy value that will be compared
        # against during validation
        self.policy_val_oracle = on_policy_estimate(
            env=self.env, pi_e=self.pi_e, gamma=self.gamma,
            num_tau=None, tau_len=None, tau_list=tau_list_oracle)

    def log(self, train_data_loader, val_data_loader, w, f, init_state_sampler,
            epoch):
        assert isinstance(init_state_sampler, AbstractInitStateSampler)
        print("Validation results for epoch %d" % epoch)

        # print W function on the fixed sample
        print("W sample values:", w(self.s_sample).view(-1).detach())

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            # calculate mean game objective
            obj_total = 0.0
            obj_norm = 0.0
            for s, a, s_prime, _ in data_loader:
                batch_size = len(s)
                s_0 = init_state_sampler.get_sample(batch_size)
                _, neg_obj = w_game_objective(w=w, f=f, s=s, a=a,
                                              s_prime=s_prime, pi_b=self.pi_b,
                                              pi_e=self.pi_e, s_0=s_0,
                                              gamma=self.gamma)
                obj_total += float(-neg_obj) * len(s)
                obj_norm += len(s)
            mean_obj = obj_total / obj_norm
            print("%s mean objective: %f" % (which, mean_obj))

            # estimate policy value
            policy_val_estimate = w_estimator(
                tau_list_data_loader=data_loader, pi_e=self.pi_e,
                pi_b=self.pi_b, w=w)
            square_error = (policy_val_estimate - self.policy_val_oracle) ** 2
            print("%s policy value estimate squared error: %f"
                  % (which, square_error))
        print("")


class SimpleDiscretePrintWLogger(SimplePrintWLogger):
    def __init__(self, env, pi_e, pi_b, gamma, oracle_tau_len=1000000):
        SimplePrintWLogger.__init__(self, env, pi_e, pi_b, gamma,
                                    oracle_tau_len=oracle_tau_len)

        # estimate oracle W vector
        self.w_oracle = calculate_tabular_w_oracle(
            env=env, pi_b=pi_b, pi_e=pi_e, gamma=gamma, num_s=env.num_s)

    def log(self, train_data_loader, val_data_loader, w, f, init_state_sampler,
            epoch):
        assert isinstance(init_state_sampler, AbstractInitStateSampler)
        print("Validation results for epoch %d" % epoch)

        # print W function on the fixed sample
        print("W sample values:", w(self.s_sample).view(-1).detach())

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            # calculate mean game objective
            obj_total = 0.0
            obj_norm = 0.0
            for s, a, s_prime, _ in data_loader:
                batch_size = len(s)
                s_0 = init_state_sampler.get_sample(batch_size)
                _, neg_obj = w_game_objective(w=w, f=f, s=s, a=a,
                                              s_prime=s_prime, pi_b=self.pi_b,
                                              pi_e=self.pi_e, s_0=s_0,
                                              gamma=self.gamma)
                obj_total += float(-neg_obj) * len(s)
                obj_norm += len(s)
            mean_obj = obj_total / obj_norm
            print("%s mean objective: %f" % (which, mean_obj))

            # calculate error in W
            w_err_total = 0.0
            w_err_norm = 0.0
            for s, a, s_prime, _ in data_loader:
                w_pred = w(s).view(-1)
                w_true = self.w_oracle[s]
                w_err_total += ((w_pred - w_true) ** 2).sum()
                w_err_norm += len(s)
            w_rmse = float((w_err_total / w_err_norm) ** 0.5)
            print("%s W RMSE: %f" % (which, w_rmse))

            # estimate policy value
            policy_val_estimate = w_estimator(
                tau_list_data_loader=data_loader, pi_e=self.pi_e,
                pi_b=self.pi_b, w=w)
            square_error = (policy_val_estimate - self.policy_val_oracle) ** 2
            print("%s policy value estimate squared error: %f"
                  % (which, square_error))
        print("")

