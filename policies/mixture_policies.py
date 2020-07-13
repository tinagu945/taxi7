from policies.discrete_policy import DiscretePolicy


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
        assert isinstance(pi_1, DiscretePolicy)
        assert isinstance(pi_2, DiscretePolicy)
        policy_probs = (pi_1_weight * pi_1.policy_probs
                        + (1 - pi_1_weight) * pi_2.policy_probs)
        DiscretePolicy.__init__(self, policy_probs)


class GenericMixturePolicy(object):
    """
    class to implement a generic mixture between two different policies
    that are not necessarily discrete... note the discrete version is designed
    to be more efficient in the discrete case
    """
    def __init__(self, pi_1, pi_2, pi_1_weight):
        """
        :param pi_1: first policy to mix between
        :param pi_2: second policy to mix between (note we assume that each of
            these policies is a valid policy from this module)
        :param pi_1_weight: weight to give to first policy
        :return: new mixture policy
        """
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.pi_1_weight = pi_1_weight

    def __call__(self, s):
        return (self.pi_1_weight * self.pi_1(s)
                + (1 - self.pi_1_weight) * self.pi_2(s))

