import torch
from dataset.tau_list_dataset import TauListDataset


def q_ERM_loss(q, pi_e, s, a, s_prime, r, gamma):
    q_of_s_a = torch.gather(q(s), dim=1, index=a.view(-1, 1)).view(-1)
    v_of_ss = (pi_e(s_prime) * q(s_prime)).sum(1).detach()
    obj = ((q_of_s_a - r - gamma * v_of_ss) ** 2).mean()
    return obj


def train_q_network_erm(train_data, pi_e, num_epochs,
                        batch_size, q, q_optimizer, gamma,
                        val_data=None, val_freq=10, logger=None,
                        q_scheduler=None):
    """
    :param train_data: list of trajectories logged from behavior policy
        to use for training
    :param pi_e: evaluation policy (should be from policies module)
    :param num_epochs: number of epochs to perform training for
    :param batch_size: batch size for alternating gradient descent
    :param q: initial q network (should be torch.nn.Module)
    :param q_optimizer: optimizer for q network
    :param gamma: discount factor (0 < gamma <= 1)
    :param val_data: (optional) list of validation trajectories for logging
    :param val_freq: frequency of how often we perform validation logging (only
        if validation data is provided)
    :param logger: (optional) logger object (should be subclass of
        AbstractQLogger)
    :return: None
    """
    assert isinstance(train_data, TauListDataset)
    train_data_loader = train_data.get_data_loader(batch_size)
    if val_data:
        val_data_loader = val_data.get_data_loader(batch_size)
    else:
        val_data_loader = None

    for epoch in range(num_epochs):
        for s, a, s_prime, r in train_data_loader:
            # calculate classical q learning objective
            obj = q_ERM_loss(q, pi_e, s, a, s_prime, r, gamma)

            q_optimizer.zero_grad()
            obj.backward()
            q_optimizer.step()

            if q_scheduler:
                q_scheduler.step()

        if logger and epoch % val_freq == 0:
            logger.log_benchmark(train_data_loader, val_data_loader, q, epoch)
