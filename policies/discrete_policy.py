import numpy as np
import torch


class DiscretePolicy(object):
    """
    class to implement a generic discrete policy, which takes either a
    single state as input and outputs a single array of probabilities,
    or takes a batch of states, and outputs a corresponding batch of
    array probabilities

    only relevant for tabular settings where states are integers,
    and is essentially just a wrapper around a simple array of policy probs
    """

    def __init__(self, policy_probs):
        """
        :param policy_probs: numpy array or pytorch tensor containing policy
            probabilities, should be of shape (num_states, num_actions)
        """
        if isinstance(policy_probs, np.ndarray):
            self.policy_probs = torch.from_numpy(policy_probs).float()
        else:
            self.policy_probs = policy_probs

    def __call__(self, s):
        if isinstance(s, torch.Tensor):
            s = s.view(-1)
        return self.policy_probs[s]

    def get_probability_table(self):
        return self.policy_probs


class EncodingDiscretePolicy(DiscretePolicy):
    """
    class to implement a policy for continuous states that have a known
    discrete encoding (e.g. for Taxi environment when we represent state
    via image-style tensor)
    """

    def __init__(self, policy_probs, encoder):
        """
        :param policy_probs: numpy array or pytorch tensor containing policy
            probabilities, should be of shape (num_states, num_actions)
        :param encoder: function that takes either a single state or a
            FloatTensor batch of states, and respectively returns either a
            single encoded state or a LongTensor batch of encoded states
        """
        DiscretePolicy.__init__(self, policy_probs)
        self.encoder = encoder

    def __call__(self, s):
        s_encoded = self.encoder(s)
        return DiscretePolicy.__call__(self, s_encoded)
