import torch


class RandomPolicy(object):
    """
    Class for a policy that just returns actions at random, useful for debugging
    """
    def __init__(self, num_a, state_rank=0):
        """
        :param num_a: number of possible actions
        :param state_rank: rank of states (if they are tensors), if states
            are integers this should be zero
        """
        self.num_a = num_a
        self.state_rank = state_rank

    def __call__(self, s):
        if isinstance(s, torch.Tensor) and len(s.shape) > self.state_rank:
            batch_size = len(s)
            return torch.ones(batch_size, self.num_a) / self.num_a
        else:
            return torch.ones(self.num_a) / self.num_a

