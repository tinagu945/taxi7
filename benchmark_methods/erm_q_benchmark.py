import torch

from dataset.tau_list_dataset import TauListDataLoader


def train_q_network_erm(train_tau_list, pi_e, num_epochs,
                        batch_size, q, q_optimizer, gamma,
                        val_tau_list=None, val_freq=10, logger=None):
    """
    :param train_tau_list: list of trajectories logged from behavior policy
        to use for training
    :param pi_e: evaluation policy (should be from policies module)
    :param num_epochs: number of epochs to perform training for
    :param batch_size: batch size for alternating gradient descent
    :param q: initial q network (should be torch.nn.Module)
    :param q_optimizer: optimizer for q network
    :param gamma: discount factor (0 < gamma <= 1)
    :param val_tau_list: (optional) list of validation trajectories for logging
    :param val_freq: frequency of how often we perform validation logging (only
        if validation data is provided)
    :param logger: (optional) logger object (should be subclass of
        AbstractQLogger)
    :return: None
    """
    train_data_loader = TauListDataLoader(tau_list=train_tau_list,
                                          batch_size=batch_size)
    if val_tau_list:
        val_data_loader = TauListDataLoader(tau_list=val_tau_list,
                                            batch_size=batch_size)
    else:
        val_data_loader = None

    for epoch in range(num_epochs):
        for s, a, s_prime, r in train_data_loader:
            # calculate classical q learning objective
            q_of_s_a = torch.gather(q(s), dim=1, index=a.view(-1, 1)).view(-1)
            v_of_ss = (pi_e(s_prime) * q(s_prime)).sum(1).detach()
            obj = ((q_of_s_a - r - gamma * v_of_ss) ** 2).mean()

            q_optimizer.zero_grad()
            obj.backward()
            q_optimizer.step()

        if logger and epoch % val_freq == 0:
            logger.log_benchmark(train_data_loader, val_data_loader, q, epoch)
