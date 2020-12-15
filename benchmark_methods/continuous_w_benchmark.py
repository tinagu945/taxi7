import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
# from torch.utils.tensorboard import SummaryWriter
from dataset.init_state_sampler import CartpoleInitStateSampler
from estimators.infinite_horizon_estimators import w_estimator
from models.continuous_models import StateClassifierModel, WOracleModel, WNetworkModel
from models.w_adversary_wrapper import WAdversaryWrapper
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from debug_logging.w_logger import SimplestWLogger
from dataset.tau_list_dataset import TauListDataset


def create_oracle_datasets(s_e, s_b, batch_size, train_ratio=0.8, thin_ratio=0.2):
    # do thinning
    skip_freq = int(1.0 / thin_ratio)
    s_e_idx = list(range(0, len(s_e), skip_freq))
    s_b_idx = list(range(0, len(s_b), skip_freq))
    s_e = s_e[s_e_idx]
    s_b = s_b[s_b_idx]

    # Input is S in SASR only
    train_len = int(train_ratio*s_e.size(0))

    train_s = torch.cat((s_e[:train_len], s_b[:train_len]))
    train_labels = torch.zeros((train_s.size(0),)).long()
    train_labels[:train_len] = 1
    train_data = TensorDataset(train_s, train_labels)
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    val_s = torch.cat((s_e[train_len:], s_b[train_len:]))
    val_labels = torch.zeros((val_s.size(0),)).long()
    val_labels[:(s_e[train_len:].size(0))] = 1
    val_data = TensorDataset(val_s, val_labels)
    val_data_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False)
    return train_data_loader, val_data_loader, val_s.size(0)


def calculate_continuous_w_oracle(env, pi_b, pi_e, gamma, hidden_dim, cuda=False, lr=1e-3,
                                  tau_len=1000000, burn_in=100000, batch_size=1024, epoch=100, num_tau=1,
                                  load=True, load_path=('tau_e_cartpole/', 'tau_b_cartpole/'), prefix=''):
    """
    :param env: environment (should be AbstractEnvironment)
    :param pi_b: behavior policy (should be from policies module)
    :param pi_e: evaluation policy (should be from policies module)
    :param gamma: discount factor
    :param num_s: number of different states
    :param tau_len: length to trajectory to use for monte-carlo estimate
    :param burn_in: burn-in period for monte-carlo sampling
    :return w: Network with same architecture as the trainining model

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf

    Action:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
    """
    if load:
        # The stored s_e is much longer than s_b since s_e was used only for estimating policy value.
        tau_list_b = TauListDataset.load(load_path[1], prefix[1])
        s_b = tau_list_b.s
        tau_list_e = TauListDataset.load(load_path[0], prefix[0])
        s_e = tau_list_e.s[:s_b.size(0)]

    else:
        tau_list_e = env.generate_roll_out(pi=pi_e, num_tau=num_tau, tau_len=tau_len,
                                           gamma=gamma, burn_in=burn_in)
        tau_list_b = env.generate_roll_out(pi=pi_b, num_tau=num_tau, tau_len=tau_len,
                                           burn_in=burn_in)
        # tau_list_e.save(load_path[0])
        # tau_list_b.save(load_path[1])
        s_e = tau_list_e.s  # [:][0]
        s_b = tau_list_b.s  # [:][0]

    x = StateClassifierModel(env.state_dim, 128, out_dim=2)
    train_data_loader, val_data_loader, val_len = create_oracle_datasets(
        s_e, s_b, batch_size, train_ratio=0.8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(x.model.parameters()), lr=lr)
    optimizer_lbfgs = optim.LBFGS(x.model.parameters())

    lowest_loss = np.inf
    for i in range(epoch):
        print(i)
        x.train()

        loss_sum_train = 0.0
        norm_train = 0.0
        num_correct_sum_train = 0
        for batch_idx, (data, labels) in enumerate(train_data_loader):
            if cuda:
                data, labels = data.cuda(), labels.cuda()
            logits = x(data)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prob = logits.softmax(-1)
            num_correct = float((prob.argmax(-1) == labels).sum())
            num_correct_sum_train += num_correct
            loss_sum_train += (float(loss.detach()) * len(data))
            norm_train += len(data)

        train_acc = num_correct_sum_train / norm_train
        avg_loss = loss_sum_train / norm_train
        print('train accuracy ', train_acc)
        print('avg loss ', avg_loss)

        if i % 1 == 0:
            x.eval()
            num_correct_sum = 0.0
            loss_sum = 0.0
            norm = 0.0
            for batch_idx, (data, labels) in enumerate(val_data_loader):
                if cuda:
                    data, labels = data.cuda(), labels.cuda()

                logits = x(data)
                prob = logits.softmax(-1)
                loss = criterion(logits, labels)
                num_correct = float((prob.argmax(-1) == labels).sum())
                num_correct_sum += num_correct
                loss_sum += (float(loss.detach()) * len(data))
                norm += len(data)
                # import pdb
                # pdb.set_trace()
                if batch_idx == 0:
                    print(prob[:50], labels[:50])

            test_acc = num_correct_sum / norm
            avg_loss = loss_sum / norm
            print('test accuracy ', test_acc)
            print('avg loss ', avg_loss)
            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
                torch.save(x.state_dict(), './continuous_w_oracle.pt')
                print('best model saved with loss ', lowest_loss)

    train_data = train_data_loader.dataset[:]

    def closure_():
        optimizer_lbfgs.zero_grad()
        s_ = train_data[0]
        labels_ = train_data[1]
        logits_ = x(s_)
        loss_ = criterion(logits_, labels_)
        loss_.backward()
        return loss_
    optimizer_lbfgs.step(closure_)

    return WOracleModel(state_classifier=x, reg=1e-7,
                        train_data_loader=train_data_loader)


def create_RKHS_dataset(tau_list_b, batch_size, train_ratio=0.8, thin_ratio=0.2):
    s_b, a_b, s_prime_b = tau_list_b.s, tau_list_b.a, tau_list_b.s_prime
    # do thinning
    skip_freq = int(1.0 / thin_ratio)
    s_b_idx = list(range(0, len(s_b), skip_freq))
    s_b = s_b[s_b_idx]
    a_b = a_b[s_b_idx]
    s_prime_b = s_prime_b[s_b_idx]

    # Input is S in SASR only
    train_len = int(train_ratio*s_b.size(0))

    train_s, train_a, train_s_prime = s_b[:
                                          train_len], a_b[:train_len], s_prime_b[:train_len]
    train_data = TensorDataset(train_s, train_a, train_s_prime)
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    val_s, val_a, val_s_prime = s_b[train_len:
                                    ], a_b[train_len:], s_prime_b[train_len:]
    # import pdb
    # pdb.set_trace()
    val_data = TensorDataset(val_s, val_a, val_s_prime)
    val_data_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_data_loader, val_data_loader, val_s.size(0)


def RKHS_method(env, pi_b, pi_e, tau_list_b, gamma, hidden_dim, init_state_sampler, kernel, logger=None, cuda=False, lr=5e-3,
                batch_size=1024, epoch=50):
    """
    kernel: a function that takes in 2 inputs and outputs their distance.
    """
    s_b = tau_list_b.s
    s_prime_b = tau_list_b.s_prime

    # import random
    # sample_idx = list(range(500))
    # random.shuffle(sample_idx)
    # sample = s_b[sample_idx[:10]]

    std = torch.std(torch.sqrt(((s_b-s_prime_b)**2).sum(1)))  # a scalar

    train_data_loader, val_data_loader, val_len = create_RKHS_dataset(
        tau_list_b, batch_size, train_ratio=0.8)
    w = WNetworkModel(env.state_dim, hidden_dim, positive_output=True)
    logger_data_loader = tau_list_b.get_data_loader(batch_size)

    # criterion=nn.MSELoss()
    optimizer = optim.Adam(list(w.model.parameters()), lr=lr)
    half = int(batch_size/2)
    init_samples = init_state_sampler.get_sample(half)

    i = 0
    Bi_std = 0
    count = 0
    while i < s_prime_b.size(0):
        if s_prime_b.size(0)-i < half:
            break
        # print(torch.std(torch.sqrt(
        #     ((s_prime_b[i:i+half]-init_samples)**2).sum(1))))
        Bi_std += torch.std(torch.sqrt(
            ((s_prime_b[i:i+half]-init_samples)**2).sum(1)))
        count += 1
        i += half
    Bi_std /= count
    # print('here', Bi_std)

    lowest_loss = np.inf
    for i in range(epoch):
        if i % 10 == 0 and logger:
            w.eval()
            logger.log_benchmark(logger_data_loader, None, w, i)
        w.train()
        losses = []
        for batch_idx, (train_s, train_a, train_s_prime) in enumerate(train_data_loader):
            s_i, a_i, s_prime_i = train_s[:half], train_a[:
                                                          half], train_s_prime[:half]
            s_j, a_j, s_prime_j = train_s[half:], train_a[half:
                                                          ], train_s_prime[half:]

            eta_i = torch.gather(pi_e(s_i) / (1e-5+pi_b(s_i)), dim=1,
                                 index=a_i.view(-1, 1)).view(-1)
            # gamma w'(S_i) eta(A_i, S_i) - w(S'_i)
            delta_i = gamma*w(s_i).squeeze()*eta_i-w(s_prime_i).squeeze()
            eta_j = torch.gather(pi_e(s_j) / (1e-5+pi_b(s_j)), dim=1,
                                 index=a_j.view(-1, 1)).view(-1)
            # gamma w'(S_i) eta(A_i, S_i) - w(S'_i)
            # import pdb
            # pdb.set_trace()
            delta_j = gamma*w(s_j).squeeze()*eta_j-w(s_prime_j).squeeze()
            first_term = (
                (delta_i*delta_j*kernel(s_prime_i, s_prime_j, std)).sum())/(half**2)

            B_i = kernel(s_prime_i, init_samples, Bi_std).sum()/half
            second_term = (2*(1-gamma)/half)*((delta_i*B_i).sum())

            loss = -(first_term+second_term)  # maximization
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(i, np.mean(losses))
        # print(w(sample))
    return w


def gaussian_kernel(a, b, std):
    l2 = torch.sqrt(((a-b)**2).sum(1))
    return np.exp(-l2**2/(2*std**2))/(std*np.sqrt(2*np.pi))


def debug_RKHS_method():
    from environments.cartpole_environment import CartpoleEnvironment
    from policies.mixture_policies import GenericMixturePolicy
    from policies.cartpole_policies import load_cartpole_policy

    gamma = 0.98
    alpha = 0.6
    temp = 2.0
    hidden_dim = 50
    pi_e_name = '456_-99501_cartpole_best'
    pi_other_name = '400_-99761_cartpole'
    load = True
    est_policy = False
    load_path = 'tau_b_cartpole/'
    prefix = pi_other_name+'_train_'

    env = CartpoleEnvironment(reward_reshape=False)
    init_state_sampler = CartpoleInitStateSampler(env)
    pi_e = load_cartpole_policy("cartpole_weights_1/"+pi_e_name+".pt", temp, env.state_dim,
                                hidden_dim, env.num_a)
    pi_other = load_cartpole_policy("cartpole_weights_1/"+pi_other_name+".pt", temp,
                                    env.state_dim, hidden_dim, env.num_a)
    pi_b = GenericMixturePolicy(pi_e, pi_other, alpha)
    if load:
        # s_b = torch.load(open(os.path.join(load_path, prefix+'s.pt'), 'rb'))
        # a_b = torch.load(open(os.path.join(load_path, prefix+'a.pt'), 'rb'))
        # s_prime_b = torch.load(
        #     open(os.path.join(load_path, prefix+'s_prime.pt'), 'rb'))
        tau_list_b = TauListDataset.load(load_path, prefix)
    else:
        tau_list_b = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                           burn_in=burn_in)
        # TODO: figure out current file and path naming rule
        tau_list_b.save(load_path)
        # s_b = tau_list_b[:][0]
        # # a_b = tau_list_b[:][1]
        # s_prime_b = tau_list_b[:][2]

    tau_e_path = 'tau_e_cartpole/'
    test_data_pi_e = TauListDataset.load(
        tau_e_path, prefix=pi_e_name+"_gamma"+str(gamma)+'_')
    policy_val_oracle = float(test_data_pi_e.r.mean().detach())
    # logger = ContinuousWLogger(env, pi_e, pi_b, gamma,
    #                            False, False, None, policy_val_oracle)
    logger = SimplestWLogger(env, pi_e, pi_b, gamma,
                             False, False, None, policy_val_oracle)

    # if load:
    #     w = load_continuous_w_oracle(
    #         env, hidden_dim, './continuous_w_oracle.pt')
    # else:
    #     w = calculate_continuous_w_oracle(
    #         env, pi_b, pi_e, gamma, hidden_dim, prefix=pi_other_name+"_train_",
    #         load=True, epoch=10)
    w = RKHS_method(env, pi_b, pi_e, tau_list_b, gamma, hidden_dim,
                    init_state_sampler, gaussian_kernel, logger=logger)


def debug_oracle():
    from environments.cartpole_environment import CartpoleEnvironment
    from policies.mixture_policies import GenericMixturePolicy
    from policies.cartpole_policies import load_cartpole_policy

    gamma = 0.98
    alpha = 0.6
    temp = 2.0
    hidden_dim = 50
    pi_e_name = '456_-99501_cartpole_best'
    pi_other_name = '400_-99761_cartpole'
    load = False
    est_policy = True

    env = CartpoleEnvironment(reward_reshape=False)
    init_state_sampler = CartpoleInitStateSampler(env)
    pi_e = load_cartpole_policy("cartpole_weights_1/"+pi_e_name+".pt", temp, env.state_dim,
                                hidden_dim, env.num_a)
    pi_other = load_cartpole_policy("cartpole_weights_1/"+pi_other_name+".pt", temp,
                                    env.state_dim, hidden_dim, env.num_a)
    pi_b = GenericMixturePolicy(pi_e, pi_other, alpha)

    if load:
        w = load_continuous_w_oracle(
            env, hidden_dim, './continuous_w_oracle.pt')
    else:
        w = calculate_continuous_w_oracle(
            env, pi_b, pi_e, gamma, hidden_dim, prefix=[
                pi_e_name+"_gamma"+str(gamma)+'_', pi_other_name+"_train_"],
            epoch=300)
        # load=False, num_tau=20, tau_len=100000, burn_in=1000,

    if est_policy:
        from estimators.infinite_horizon_estimators import oracle_w_estimator
        from estimators.benchmark_estimators import on_policy_estimate
        tau_e_path = 'tau_e_cartpole/'
        tau_b_path = 'tau_b_cartpole/'
        tau_len = 200000
        burn_in = 100000
        test_data = TauListDataset.load(tau_b_path, pi_other_name+'_train_')
        test_data_pi_e = TauListDataset.load(
            tau_e_path, prefix=pi_e_name+"_gamma"+str(gamma)+'_')
        test_data_loader = test_data.get_data_loader(1024)
        policy_val_estimate = w_estimator(tau_list_data_loader=test_data_loader,
                                          pi_e=pi_e, pi_b=pi_b, w=w)
        # policy_val_estimate = oracle_w_estimator(
        #     tau_list_data_loader=test_data_loader, pi_e=pi_e,
        #     pi_b=pi_b, w_oracle=w)

        # policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
        #                                        num_tau=1, tau_len=1000000,
        #                                        load_path=tau_e_path)
        policy_val_oracle = float(test_data_pi_e.r.mean().detach())
        squared_error = (policy_val_estimate - policy_val_oracle) ** 2
        print('W orcacle estimates & true policy value ',
              policy_val_estimate, policy_val_oracle)
        print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    seed = 8
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    debug_RKHS_method()
    # debug_oracle()
