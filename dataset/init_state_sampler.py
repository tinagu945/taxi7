import numpy as np
import torch


class AbstractInitStateSampler(object):
    def __init__(self):
        pass

    def get_sample(self, batch_size):
        raise NotImplementedError()

    def compute_mean(self, f):
        """
        method that takes a function that operates on states, and either
        computes or estimates the mean of the function using the sampler
        :param f: function that takes a state or batch of states as input,
            and returns a float or FloatTensor as output
        """
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
        self.init_state_dist = init_state_dist
        self.states = torch.LongTensor(range(len(p)))
        self.init_state_dist_np = p / p.sum()
        self.states_np = np.array(range(len(p)))

    def get_sample(self, batch_size):
        init_batch = np.random.choice(self.states_np, size=batch_size,
                                      p=self.init_state_dist_np)
        return torch.from_numpy(init_batch)

    def compute_mean(self, f):
        return (f(self.states) * self.init_state_dist).sum()


class DecodingDiscreteInitStateSampler(DiscreteInitStateSampler):
    """
    This class is for environments where states are continuous but have
    a discrete encoding (such as with Taxi environment with continuous states)
    """
    def __init__(self, init_state_dist, decoder):
        """
        :param init_state_dist: FloatTensor of shape (num_s,), where num_s
            is number of states, containing the probabilities of each state
            at t=0
        :param decoder: function that takes either a single state encoding
            or a LongTensor batch of state encodings, and respectively returns
            either a single decoded state or a batch of decoded states
        :param decoder:
        :return:
        """
        DiscreteInitStateSampler.__init__(self, init_state_dist)
        self.decoder = decoder
        self.states = decoder(self.states)

    def get_sample(self, batch_size):
        s_encoded = DiscreteInitStateSampler.get_sample(self, batch_size)
        return self.decoder(s_encoded)
