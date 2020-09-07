import random
import torch
import os
import datetime
from adversarial_learning.game_objectives import q_game_objective
from benchmark_methods.discrete_q_benchmark import fit_q_tabular
from estimators.benchmark_estimators import on_policy_estimate
from estimators.infinite_horizon_estimators import q_estimator
from models.continuous_models import QOracleModel
from dataset.tau_list_dataset import TauListDataset


class AbstractQLogger(object):
    def __init__(self, env, pi_e, gamma, save_model):
        self.env = env
        self.pi_e = pi_e
        self.gamma = gamma
        self.save_model = save_model
        if self.save_model:
            self.lowest_err = float('inf')
            now = datetime.datetime.now()
            self.path = 'logs/' + str(now.isoformat())

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

    def save_q(self, square_error, q, epoch, suffix=''):
        if square_error < self.lowest_err:
            print('New best! Square_error is', square_error, epoch)
            self.lowest_err = square_error
            with open(os.path.join(self.path, 'best_q_'+suffix+'.pt'), 'wb') as f:
                torch.save(q.model.state_dict(), f)
        else:
            with open(os.path.join(self.path, str(epoch)+'_q_'+suffix+'.pt'), 'wb') as f:
                torch.save(q.model.state_dict(), f)
        print('Model saved in ', self.path)


class QLogger(AbstractQLogger):
    def __init__(self, env, pi_e, gamma, init_state_sampler, save_model, tensorboard, oracle_tau_len=100000, load_path=None):
        AbstractQLogger.__init__(self, env, pi_e, gamma, save_model)

        # train an oracle Q function that will be used for debug_logging error
        # in Q function
        sample_idx = list(range(oracle_tau_len))
        random.shuffle(sample_idx)
        if load_path:
            print('Loading datasets for q logger...')
            pi_e_data = TauListDataset.load(load_path)
        else:
            print(
                'Logger not loading oracle e data, so generating pi_e of length ', oracle_tau_len)
            pi_e_data = self.env.generate_roll_out(
                pi=self.pi_e, num_tau=1, tau_len=oracle_tau_len, gamma=gamma)

        self.s_sample = pi_e_data.s[sample_idx[:5]]
        self.a_sample = pi_e_data.a[sample_idx[:5]]
        self.s_prime_sample = pi_e_data.s_prime[sample_idx[:5]]
        self.r_sample = pi_e_data.r[sample_idx[:5]]
        self.policy_val_oracle = -40.6  # on_policy_estimate(
        # env=env, pi_e=pi_e, gamma=gamma, num_tau=1, tau_len=1000000)
        print('Policy_val_oracle', self.policy_val_oracle)
        # calculate oracle estimate of policy value that will be compared
        # against during validation
        self.init_state_sampler = init_state_sampler
        self.tensorboard = tensorboard
        if self.tensorboard:
            # Make sure we create a folder for tensorboard.
            if not self.save_model:
                now = datetime.datetime.now()
                self.path = 'logs/' + str(now.isoformat())

            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.path)

    def log_benchmark(self, train_data_loader, val_data_loader, q, epoch):
        print("[log_benchmark] Validation results for benchmark epoch %d" % epoch)

        # print Q function on the fixed sample
        print("[log_benchmark] Q function sample values:")
        print(q(self.s_sample).detach())

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
            print("[log_benchmark] %s Q function RMSE: %f" % (which, q_rmse))
            if self.tensorboard and which == 'Val':
                self.writer.add_scalar(
                    'benchmark_Q_RMSE_'+which, q_rmse, epoch)

        q_of_s_a_sample = torch.gather(q(self.s_sample), dim=1,
                                       index=self.a_sample.view(-1, 1)).view(-1)
        v_of_ss_sample = (self.pi_e(self.s_prime_sample)
                          * q(self.s_prime_sample)).sum(1)
        print("[log_benchmark] v(ss) sample:", v_of_ss_sample.detach())
        print("[log_benchmark] q(s,a) sample:", q_of_s_a_sample.detach())
        print("[log_benchmark] eq sample:", (self.r_sample + self.gamma * v_of_ss_sample
                                             - q_of_s_a_sample).detach())
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
        print("[log_benchmark] mean eq:", mean_eq)
        print("[log_benchmark] uniform gmm norm:",
              (mean_eq ** 2) / mean_eq_squared)

        # estimate policy value
        policy_val_estimate = q_estimator(
            pi_e=self.pi_e, gamma=self.gamma, q=q,
            init_state_sampler=self.init_state_sampler)
        square_error = (policy_val_estimate - self.policy_val_oracle) ** 2
        print('[log_benchmark] Policy_val_oracle', self.policy_val_oracle)
        print("[log_benchmark] Policy value estimate squared error:", square_error)
        print("")

        if self.tensorboard:
            self.writer.add_scalar('benchmark_mean_Q_eq', mean_eq, epoch)
            self.writer.add_scalar('benchmark_uniform_Q_gmm_norm',
                                   (mean_eq ** 2) / mean_eq_squared, epoch)
            self.writer.add_scalar(
                'benchmark_Q_policy_sqrerr', square_error, epoch)

        if self.save_model:
            self.save_q(square_error, q, epoch, suffix='ERM')

    def log(self, train_data_loader, val_data_loader, q, f, epoch):
        print("Validation results for epoch %d" % epoch)

        # print Q function on the fixed sample
        print("Q function sample values:")
        print(q(self.s_sample).detach())

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

            if self.tensorboard:
                self.writer.add_scalar('mean_Q_obj_'+which, mean_obj, epoch)
                self.writer.add_scalar('Q_RMSE_'+which, q_rmse, epoch)

        q_of_s_a_sample = torch.gather(q(self.s_sample), dim=1,
                                       index=self.a_sample.view(-1, 1)).view(-1)
        f_of_s_a_sample = torch.gather(f(self.s_sample), dim=1,
                                       index=self.a_sample.view(-1, 1)).view(-1)
        v_of_ss_sample = (self.pi_e(self.s_prime_sample)
                          * q(self.s_prime_sample)).sum(1)
        print("v(ss) sample:", v_of_ss_sample.detach())
        print("q(s,a) sample:", q_of_s_a_sample.detach())
        print("f(s,a) sample:", f_of_s_a_sample.detach())
        print("eq sample:", (self.r_sample + self.gamma * v_of_ss_sample
                             - q_of_s_a_sample).detach())
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
        print('Policy_val_oracle', self.policy_val_oracle)
        print("Policy val ue estimate squared error:", square_error)

        if self.tensorboard:
            self.writer.add_scalar('mean_Q_eq', mean_eq, epoch)
            self.writer.add_scalar('uniform_Q_gmm_norm',
                                   (mean_eq ** 2) / mean_eq_squared, epoch)
            self.writer.add_scalar('policy_Q_sqrterr', square_error, epoch)
        if self.save_model:
            self.save_q(square_error, q, epoch)
        print("")


class DiscreteQLogger(QLogger):
    def __init__(self, env, pi_e, pi_b, gamma, init_state_sampler, save_model, tensorboard, oracle_tau_len=1000000):
        QLogger.__init__(self, env, pi_e, pi_b, gamma, init_state_sampler,
                         save_model, oracle_tau_len=oracle_tau_len)
        # estimate oracle W vector
        self.q_oracle = fit_q_tabular(data=pi_e_data, pi=self.pi_e,
                                      gamma=self.gamma)


class ContinuousQLogger(QLogger):
    def __init__(self, env, pi_e, pi_b, gamma, init_state_sampler, save_model, tensorboard, oracle_path, oracle_tau_len=1000000, load_path=None):
        QLogger.__init__(self, env, pi_e, gamma, init_state_sampler,
                         save_model, tensorboard, oracle_tau_len=oracle_tau_len, load_path=load_path)
        # estimate oracle W vector
        self.q_oracle = QOracleModel.load_continuous_q_oracle(
            env, 50, env.num_a, oracle_path)
