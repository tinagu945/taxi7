import random
import torch
import os
import datetime
from adversarial_learning.game_objectives import w_game_objective
from dataset.init_state_sampler import AbstractInitStateSampler
from benchmark_methods.discrete_w_oracle_benchmark import \
    calculate_tabular_w_oracle
# from estimators.benchmark_estimators import _estimateon_policy
from estimators.infinite_horizon_estimators import w_estimator
from models.continuous_models import \
    load_continuous_w_oracle


class AbstractWLogger(object):
    def __init__(self, env, pi_e, pi_b, gamma, save_model):
        self.env = env
        self.pi_e = pi_e
        self.pi_b = pi_b
        self.gamma = gamma
        self.save_model = save_model
        if self.save_model:
            self.lowest_err = float('inf')
            now = datetime.datetime.now()
            self.path = 'logs/' + str(now.isoformat())

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

    def save_w(self, square_error, w, epoch):
        if square_error < self.lowest_err:
            print('New best! ', square_error, epoch)
            self.lowest_err = square_error
            with open(os.path.join(self.path, 'best_w.pt'), 'wb') as f:
                torch.save(w.state_dict(), f)
        else:
            with open(os.path.join(self.path, str(epoch)+'_w.pt'), 'wb') as f:
                torch.save(w.state_dict(), f)
        print('Model saved in ', self.path)


class SimplePrintWLogger(AbstractWLogger):
    def __init__(self, env, pi_e, pi_b, gamma, save_model,  oracle_tau_len=1000000, load_path=None):
        AbstractWLogger.__init__(self, env, pi_e, pi_b, gamma, save_model)

        sample_idx = list(range(oracle_tau_len))
        random.shuffle(sample_idx)

        if load_path:
            print('Loading datasets for w logger...')
            s_e = torch.load(open(os.path.join(load_path, 's.pt'), 'rb'))
            self.s_sample = s_e[sample_idx[:5]]
            r_e = torch.load(open(os.path.join(load_path, 'r.pt'), 'rb'))
            self.policy_val_oracle = float(r_e.mean())
        else:
            print(
                'Logger not loading oracle e data, so generating pi_e of length ', oracle_tau_len)
            # estimate oracle W vector
            pi_e_data = self.env.generate_roll_out(
                pi=self.pi_e, num_tau=1, tau_len=oracle_tau_len, gamma=gamma)
            self.s_sample = pi_e_data.s[sample_idx[:5]]
            # calculate oracle estimate of policy value that will be compared
            # against during validation
            self.policy_val_oracle = float(pi_e_data.r.mean())
        print('Logger policy_val_oracle ', self.policy_val_oracle)

    def log(self, train_data_loader, val_data_loader, w, f, init_state_sampler,
            epoch):
        assert isinstance(init_state_sampler, AbstractInitStateSampler)
        print("Validation results for epoch %d" % epoch)

        # print W function on the fixed sample
        print("W sample values:", w(self.s_sample).view(-1).detach())

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            self.log_obj_err_save(which, data_loader, w,
                                  f, init_state_sampler, epoch)
        print("")

    def log_obj_err_save(self, which, data_loader, w, f, init_state_sampler, epoch):
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
        print("%s policy value estimate %f squared error: %f"
              % (which, policy_val_estimate, square_error))

        if which == 'Val' and self.save_model:
            self.save_w(square_error, w, epoch)
        return mean_obj, policy_val_estimate, square_error

    def log_benchmark(self, train_data_loader, val_data_loader, w, epoch):
        print("Benchmark validation results for epoch %d" % epoch)

        # print W function on the fixed sample
        print("W sample values:", w(self.s_sample).view(-1).detach())

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            # estimate policy value
            policy_val_estimate = w_estimator(
                tau_list_data_loader=data_loader, pi_e=self.pi_e,
                pi_b=self.pi_b, w=w)
            square_error = (policy_val_estimate - self.policy_val_oracle) ** 2
            print("%s policy value %f estimate squared error: %f"
                  % (which, policy_val_estimate, square_error))
        print("")


class DiscreteWLogger(SimplePrintWLogger):
    def __init__(self, env, pi_e, pi_b, gamma, save_model, tensorboard, oracle_tau_len=1000000):
        SimplePrintWLogger.__init__(self, env, pi_e, pi_b, gamma, save_model,
                                    oracle_tau_len=oracle_tau_len)

        # estimate oracle W vector
        self.w_oracle = calculate_tabular_w_oracle(
            env=env, pi_b=pi_b, pi_e=pi_e, gamma=gamma, num_s=env.num_s)

        self.tensorboard = tensorboard
        if self.tensorboard:
            # Make sure we create a folder for tensorboard.
            if not self.save_model:
                now = datetime.datetime.now()
                self.path = 'logs/' + str(now.isoformat())

            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.path)

    def log(self, train_data_loader, val_data_loader, w, f, init_state_sampler,
            epoch):
        assert isinstance(init_state_sampler, AbstractInitStateSampler)
        print("Validation results for epoch %d" % epoch)

        # print W function on the fixed sample
        print("W sample values:", w(self.s_sample).view(-1).detach())

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            mean_obj, policy_val_estimate, square_error = self.log_obj_err_save(
                which, data_loader, w, f, init_state_sampler, epoch)

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

            if self.tensorboard:
                self.writer.add_scalar('mean_obj_'+which, mean_obj, epoch)
                self.writer.add_scalar('W_RMSE_'+which, w_rmse, epoch)
                self.writer.add_scalar(
                    'policy_sqrterr_'+which, square_error, epoch)
        print("")

    def log_benchmark(self, train_data_loader, val_data_loader, w, epoch):
        """
        Currently only used for ERM logging.
        """
        print("Benchmark validation results for epoch %d" % epoch)

        # print W function on the fixed sample
        print("W sample values:", w(self.s_sample).view(-1).detach())

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
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

            if self.tensorboard:
                self.writer.add_scalar(
                    'benchmark_W_RMSE_'+which, w_rmse, epoch)
                self.writer.add_scalar(
                    'benchmark_policy_sqrterr_'+which, square_error, epoch)
        print("")


class ContinuousWLogger(SimplePrintWLogger):
    def __init__(self, env, pi_e, pi_b, gamma, hidden_dim, save_model, tensorboard, oracle_path='./continuous_w_oracle.pt', oracle_tau_len=1000000, load_path='tau_e_cartpole/'):
        SimplePrintWLogger.__init__(self, env, pi_e, pi_b, gamma, save_model,
                                    oracle_tau_len=oracle_tau_len, load_path=load_path)
        # estimate oracle W vector
        self.w_oracle = load_continuous_w_oracle(env, hidden_dim, oracle_path)
        self.tensorboard = tensorboard
        if self.tensorboard:
            # Make sure we create a folder for tensorboard.
            if not self.save_model:
                now = datetime.datetime.now()
                self.path = 'logs/' + str(now.isoformat())

            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.path)

    def log(self, train_data_loader, val_data_loader, w, f, init_state_sampler,
            epoch):
        assert isinstance(init_state_sampler, AbstractInitStateSampler)
        print("Validation results for epoch %d" % epoch)

        # print W function on the fixed sample
        print("W sample values:", w(self.s_sample).view(-1).detach())

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            mean_obj, policy_val_estimate, square_error = self.log_obj_err_save(
                which, data_loader, w, f, init_state_sampler, epoch)

            # calculate error in W
            w_err_total = 0.0
            w_err_norm = 0.0
            for s, a, s_prime, _ in data_loader:
                w_pred = w(s).view(-1)
                b_prob = self.w_oracle(s).softmax(-1)[:, -1]
                w_true = (1-b_prob)/b_prob
                w_err_total += ((w_pred - w_true) ** 2).sum()
                w_err_norm += len(s)
            w_rmse = float((w_err_total / w_err_norm) ** 0.5)
            print("%s W RMSE: %f" % (which, w_rmse))

            if self.tensorboard:
                self.writer.add_scalar('mean_obj_'+which, mean_obj, epoch)
                self.writer.add_scalar('W_RMSE_'+which, w_rmse, epoch)
                self.writer.add_scalar(
                    'policy_sqrterr_'+which, square_error, epoch)
        print("")

    def log_benchmark(self, train_data_loader, val_data_loader, w, epoch):
        """
        Currently only used for ERM logging.
        """
        print("Benchmark validation results for epoch %d" % epoch)

        # print W function on the fixed sample
        print("W sample values:", w(self.s_sample).view(-1).detach())

        for which, data_loader in (("Train", train_data_loader),
                                   ("Val", val_data_loader)):
            # calculate error in W
            w_err_total = 0.0
            w_err_norm = 0.0
            for s, a, s_prime, _ in data_loader:
                w_pred = w(s).view(-1)
                b_prob = self.w_oracle(s).softmax(-1)[:, -1]
                w_true = (1-b_prob)/b_prob
                w_err_total += ((w_pred - w_true) ** 2).sum()
                w_err_norm += len(s)
            w_rmse = float((w_err_total / w_err_norm) ** 0.5)
            print("%s W RMSE: %f" % (which, w_rmse))

            # estimate policy value
            policy_val_estimate = w_estimator(
                tau_list_data_loader=data_loader, pi_e=self.pi_e,
                pi_b=self.pi_b, w=w)
            square_error = (policy_val_estimate - self.policy_val_oracle) ** 2
            print("%s policy value %f estimate squared error: %f"
                  % (which, policy_val_estimate, square_error))

            if self.tensorboard:
                self.writer.add_scalar(
                    'benchmark_W_RMSE_'+which, w_rmse, epoch)
                self.writer.add_scalar(
                    'benchmark_policy_sqrterr_'+which, square_error, epoch)
        print("")
