import numpy as np
import torch


class AbstractInitStateSampler(object):
    def __init__(self):
        pass

    def get_sample(self, batch_size):
        raise NotImplementedError()


class DiscreteInitStateSampler(AbstractInitStateSampler):
    """
    Implementation for discrete scenarios where we know the initial state
    probabilities
    """
    def __init__(self, init_state_dist):
        """
        :param init_state_dist: FloatTensor of shape (num_s,), where num_s
            is number of states, containing the probabilities of each state
            at t=0
        :return:
        """
        AbstractInitStateSampler.__init__(self)
        p = init_state_dist.numpy()
        self.init_state_dist = p / p.sum()
        self.states = np.array(range(len(p)))

    def get_sample(self, batch_size):
        init_batch = np.random.choice(self.states, size=batch_size,
                                      p=self.init_state_dist)
        return torch.from_numpy(init_batch)

