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
        :param policy_probs: numpy array of pytorch tensor containing policy
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


class MixtureDiscretePolicy(DiscretePolicy):
    """
    class to implement a policy as a mixture between two discrete policies
    useful for taxi experiments
    """
    def __init__(self, pi_1, pi_2, pi_1_weight):
        """
        :param pi_1: first policy to mix between
        :param pi_2: second policy to mix between
        :param pi_1_weight: weight to give to first policy
        :return: new mixture policy
        """

        policy_probs = (pi_1_weight * pi_1.policy_probs
                        + (1 - pi_1_weight) * pi_2.policy_probs)
        DiscretePolicy.__init__(self, policy_probs)

