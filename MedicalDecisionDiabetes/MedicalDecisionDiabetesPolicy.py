"""
Policy class -  Medical Decisions Diabetes Treatments
"""
from collections import namedtuple
import math
import random
import numpy as np

class MDDMPolicy():
    """
    Base class for Medical Decisions Diabetes Model policy
    """

    def __init__(self, model, policy_names,seed=1456897):
        """
        Initializes the policy
        :param policy_names: list(str) - list of policies
        :param model: the model that the policy is being implemented on
        """
        self.model = model
        self.policy_names = policy_names
        self.Policy = namedtuple('Policy', policy_names)
        self.seed = seed
        self.prng = np.random.RandomState(seed)

    def build_policy(self, info):
        # this function builds the policies depending on the parameters provided
        return self.Policy(*[info[k] for k in self.policy_names])


    def UCB(self, model_curr, theta):
        # this method implements the Upper Confidence Bound policy
        # N.B: can't implement this at time t=0 (from t=1 at least). 
        # Also can't divide by zero, which means we need each drug to have been tested at least once.
        #
        # Note that state has a list of 3 entries, for each key(type of drug) in the dictionary
        # {"drug" : [mu_empirical, beta, number of times drug given to patient]}
        

        aux_stats = {key:[getattr(model_curr.state, key)[0],math.sqrt(math.log(model_curr.t + 1)/(getattr(model_curr.state, key)[2] + 1))] for key in model_curr.state_names}
        stats = {key:aux_stats[key][0]+theta*aux_stats[key][1] for key in model_curr.state_names}

        
        optimal_decision = max(stats, key=stats.get)
        # print(aux_stats[optimal_decision])
        return optimal_decision

    def IE(self, model_curr, theta):
        # This method implements the Interval Estimation policy
        stats = {key:getattr(model_curr.state, key)[0]+theta/math.sqrt(getattr(model_curr.state, key)[1]) for key in model_curr.state_names}


        optimal_decision = max(stats, key=stats.get)
        return optimal_decision
    
    def PureExploitation(self, model_curr, theta):
        # This method implements the Pure Exploitation policy (theta = 0)
        stats = {key:getattr(model_curr.state, key)[0] for key in model_curr.state_names}
        
        optimal_decision = max(stats, key=stats.get)
        return optimal_decision
        
    def PureExploration(self, model_curr, theta):
        # This method implements the Pure Exploration policy (random drug every time)
        stats = {key:getattr(model_curr.state, key)[0] for key in model_curr.state_names}
        optimal_decision = self.prng.choice(list(stats))
        return optimal_decision
    