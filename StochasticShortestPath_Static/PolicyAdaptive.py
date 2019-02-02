import numpy as np

from collections import (namedtuple, defaultdict)

class Policy():
    """
    Base class for Static Stochastic Shortest Path Model policy
    """

    def __init__(self, model, policy_names):
        """
        Initializes the policy
        :param policy_names: list(str) - list of policies
        :param model: the model that the policy is being implemented on
        """
        self.model = model
        self.policy_names = policy_names
        self.Policy = namedtuple('Policy', policy_names)

    def build_policy(self, info):
        # this function builds the policies depending on the parameters provided
        return self.Policy(*[info[k] for k in self.policy_names])


    def make_decision(self,M):
        
    
        i = M.state.CurrentNode

        costs = {j:M.state.CurrentNodeLinksCost[j] + M.V_t[j] for j in M.g.edges[i]}

        optimal_decision = min(costs, key=costs.get)
        
        return optimal_decision, costs[optimal_decision]
    
    