import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from estimators.infinite_horizon_estimators import w_estimator
from models.continuous_models import StateClassifierModel, WOracleModel
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


def create_datasets(s_e, s_b, batch_size, train_ratio=0.8, thin_ratio=0.2):
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
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, val_data_loader, val_s.size(0)


def calculate_continuous_w_oracle(env, pi_b, pi_e, gamma, hidden_dim, cuda=False, lr=5e-3,
                                  tau_len=1000000, burn_in=100000, batch_size=1024, epoch=100,
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
#     tau_len=10
#     burn_in=1
    if load:
        s_e = torch.load(open(os.path.join(load_path[0], 'gamma_s.pt'), 'rb'))
        s_b = torch.load(open(os.path.join(load_path[1], prefix+'s.pt'), 'rb'))
    else:
        tau_list_e = env.generate_roll_out(pi=pi_e, num_tau=1, tau_len=tau_len,
                                           gamma=gamma, burn_in=burn_in)
        tau_list_b = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                           burn_in=burn_in)
        tau_list_e.save(load_path[0])
        tau_list_b.save(load_path[1])
        s_e = tau_list_e[:][0]
        s_b = tau_list_b[:][0]

    # import pdb
    # pdb.set_trace()
    x = StateClassifierModel(env.state_dim, 128, out_dim=2)
    train_data_loader, val_data_loader, val_len = create_datasets(
        s_e, s_b, batch_size, train_ratio=0.8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(x.model.parameters()), lr=lr)
    optimizer_lbfgs = optim.LBFGS(x.model.parameters())

    lowest_loss = np.inf
    for i in range(epoch):
        print(i)
        x.train()

        for batch_idx, (data, labels) in enumerate(train_data_loader):
            if cuda:
                data, labels = data.cuda(), labels.cuda()
            logits = x(data)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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


def debug():
    from environments.cartpole_environment import CartpoleEnvironment
    from policies.mixture_policies import GenericMixturePolicy
    from policies.cartpole_policies import load_cartpole_policy

    gamma = 0.98
    alpha = 0.6
    temp = 2.0
    hidden_dim = 50
    pi_other_name = 'cartpole_180_-99900.0'
    load = False
    est_policy = True

    env = CartpoleEnvironment()
    pi_e = load_cartpole_policy("cartpole_weights/cartpole_best.pt", temp, env.state_dim,
                                hidden_dim, env.num_a)
    pi_other = load_cartpole_policy("cartpole_weights/"+pi_other_name+".pt", temp,
                                    env.state_dim, hidden_dim, env.num_a)
    pi_b = GenericMixturePolicy(pi_e, pi_other, alpha)

    if load:
        w = load_continuous_w_oracle(
            env, hidden_dim, './continuous_w_oracle.pt')
    else:
        w = calculate_continuous_w_oracle(
            env, pi_b, pi_e, gamma, hidden_dim, prefix=pi_other_name+"_train_",
            load=True, epoch=10)

    if est_policy:
        from dataset.tau_list_dataset import TauListDataset
        from estimators.infinite_horizon_estimators import oracle_w_estimator
        from estimators.benchmark_estimators import on_policy_estimate
        tau_e_path = 'tau_e_cartpole/'
        tau_b_path = 'tau_b_cartpole/'
        tau_len = 200000
        burn_in = 100000
        test_data = TauListDataset.load(tau_b_path, pi_other_name+'_test_')
        test_data_pi_e = TauListDataset.load(tau_e_path, prefix="gamma_")
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
    debug()
