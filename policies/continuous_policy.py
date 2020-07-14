class QNetworkPolicy(object):
    """
    Policy for taking actions given some Q network, using Boltzmann softmax
    probabilities given a temperature parameter
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


