import numpy as np
import torch


class ContinuousPolicy(object):
    """
    class to implement a generic discrete policy, which takes either a
    single state as input and outputs a single array of probabilities,
    or takes a batch of states, and outputs a corresponding batch of
    array probabilities

    only relevant for tabular settings where states are integers,
    and is essentially just a wrapper around a simple array of policy probs
    """
    def __init__(self, q_network, temp):
        """
        :param policy_probs: numpy array of pytorch tensor containing policy
            probabilities, should be of shape (num_states, num_actions)
        """    
        self.q_network = q_network
        self.temp = temp

    def __call__(self, s):
        return (self.q_network(s)/self.temp).softmax(-1)

    def get_q_network(self):
        return self.q_network


class MixtureContinuousPolicy(object):
    """
    Not connected to ContinuousPolicy!
    class to implement a policy as a mixture between two continuous policies
    useful for catpole experiments.
    """
    def __init__(self, pi_1, pi_2, alpha):
        """
        :param pi_1: first NN to mix between
        :param pi_2: second NN to mix between
        :param alpha: weight to give to first policy
        :return: new mixture policy
        """
        assert isinstance(pi_1, ContinuousPolicy)
        assert isinstance(pi_2, ContinuousPolicy)
        assert pi_1.temp == pi_2.temp
        
        self.q_network_1 = pi_1.q_network
        self.q_network_2 = pi_2.q_network
        self.alpha = alpha
        self.temp = pi_1.temp

    def __call__(self, s):
        # For continuous policies, they can only be mixed as output. 
        # And the networks' outputs need to be normalized
        return self.alpha*((self.q_network_1(s)/self.temp).softmax(-1))+\
    (1-self.alpha)*((self.q_network_2(s)/self.temp).softmax(-1))
    
    def get_q_network(self):
        return [self.q_network_1, self.q_network_2]

