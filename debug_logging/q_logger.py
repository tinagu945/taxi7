import random
from adversarial_learning.game_objectives import q_game_objective
from benchmark_methods.discrete_q_benchmark import fit_q_tabular
from estimators.benchmark_estimators import on_policy_estimate
from estimators.discrete_estimators import q_estimator_discrete


class AbstractQLogger(object):
    def __init__(self, env, pi_e, gamma):
        self.env = env
        self.pi_e = pi_e
        self.gamma = gamma

    def log(self, train_data_loader, val_data_loader, q, f, epoch):
        """
        perform single log for Q learning, free to do any debug_logging as required

        :param val_data_loader: data loader for validation data
        :param q: current q function
        :param f: current adversary function
        :param epoch: current epoch number
        :return: None
        """
        raise NotImplementedError()


class SimplePrintQLogger(AbstractQLogger):
    def __init__(self, env, pi_e, gamma, init_state_dist,
                 oracle_tau_len=1000000):
        AbstractQLogger.__init__(self, env, pi_e, gamma)

        # train an oracle Q function that will be used for debug_logging error in
        # Q function
        tau_list_oracle = self.env.generate_roll_out(
            pi=self.pi_e, num_tau=1, tau_len=oracle_tau_len, gamma=gamma)
        self.q_oracle = fit_q_tabular(tau_list=tau_list_oracle, pi=self.pi_e,
                                      gamma=self.gamma)
        sample_idx = list(range(oracle_tau_len))
        random.shuffle(sample_idx)
        self.s_sample = tau_list_oracle[0][0][sample_idx[:5]]

        # calculate oracle estimate of policy value that will be compared
        # against during validation
        self.policy_val_oracle = on_policy_estimate(
            env=self.env, pi_e=self.pi_e, gamma=self.gamma,
            num_tau=None, tau_len=None, tau_list=tau_list_oracle)
        self.init_state_dist = init_state_dist

    def log(self, train_data_loader, val_data_loader, q, f, epoch):
        print("Validation results for epoch %d" % epoch)

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            # print Q function on the fixed sample
            print("Q function sample values:")
            print(q(self.s_sample))
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

        # estimate policy value
        policy_val_estimate = q_estimator_discrete(
            pi_e=self.pi_e, gamma=self.gamma, q=q,
            init_state_dist=self.init_state_dist)
        square_error = (policy_val_estimate - self.policy_val_oracle) ** 2
        print("Policy value estimate squared error:", square_error)
        print("")

