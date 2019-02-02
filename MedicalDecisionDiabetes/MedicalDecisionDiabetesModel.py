"""
Model class - Medical Decisions Diabetes Treatments

"""
from collections import namedtuple
import numpy as np
import pandas as pd
import math
from random import randint

# This function returns the precision (beta), given the s.d. (sigma)
def Beta(sigma):
    return 1/sigma**2

class MedicalDecisionDiabetesModel():
    """
    Base class for Medical Decisions Diabetes Treatments model
    """

    def __init__(self, state_names, x_names, S0, additional_params, exog_info_fn=None, transition_fn=None, objective_fn=None, seed=20180529):
        """
        Initializes the model
        
        :param state_variable: list(str) - state variable dimension names
        :param decision_variable: list(str) - decision variable dimension names
        :param state_0: dict - needs to contain at least the information to populate initial state using state_names
        :param exog_info_fn: function - calculates relevant exogenous information
        :param transition_fn: function - takes in decision variables and exogenous information to describe how the state
               evolves
        :param objective_fn: function - calculates contribution at time t
        :param seed: int - seed for random number generator
        """


        self.init_args = {seed: seed}
        self.prng = np.random.RandomState(seed)
        self.init_state = {x:[S0.loc[x, 'mu_0'],Beta(S0.loc[x, 'sigma_0']),0] for x in x_names }
        self.state_names = state_names
        self.x_names = x_names
        self.State = namedtuple('State', state_names)
        self.state = self.build_state(self.init_state)
        self.Decision = namedtuple('Decision', x_names)
        self.obj = 0.0
        self.obj_sum = 0.0
        self.sigma_w = additional_params.loc['sigma_w', 0]
        self.truth_params_dict = {} #changed later
        self.truth_type = additional_params.loc['truth_type', 0]
        self.mu = {} #updated using "exog_info_sample_mu" at the beginning of each sample path 
        self.t = 0 # time counter (in months)


        if self.truth_type == 'fixed_uniform':
            self.truth_params_dict = {x:[S0.loc[x, 'mu_fixed'],S0.loc[x, 'fixed_uniform_a'],S0.loc[x, 'fixed_uniform_b']] for x in self.x_names }
        elif self.truth_type == 'prior_uniform':
            self.truth_params_dict = {x:[S0.loc[x, 'mu_0'],S0.loc[x, 'prior_mult_a'],S0.loc[x, 'prior_mult_b']] for x in self.x_names }
        else:
            self.truth_params_dict = {x:[S0.loc[x, 'mu_truth'],S0.loc[x, 'sigma_truth'],0] for x in self.x_names }

        


    def printState(self):
        print("Current state ")
        for x in self.x_names:
            print("Treatment {}: mu {:.2f}, sigma {:.2f} and N {}".format(x,getattr(self.state,x)[0],1/math.sqrt(getattr(self.state,x)[1]),getattr(self.state,x)[2]))
        print("\n\n")
    

    def printTruth(self):
        print("Model truth_type {}. Meaurement noise sigma_W {} ".format(self.truth_type,self.sigma_w))
        for x in self.x_names:
            print("Treatment {}: par1 {:.2f}, par2 {:.2f} and par3 {}".format(x,self.truth_params_dict[x][0],self.truth_params_dict[x][1],self.truth_params_dict[x][2]))
        print("\n\n")
    

    # this function gives a state containing all the state information needed
    # State is a vector of dim.5 where each entry is a list: [mu, beta]
    # want to replace the attribute of a state corresponding to the decision
    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    # this function gives a decision 
    # Our decision will be the choice of medication to try for a month. 
    # N.B: This function is currently not in use.
    def build_decision(self, info):
        return self.Decision(*[info[k] for k in self.x_names])


    def exog_info_sample_mu(self):
        if self.truth_type == "known":
            self.mu = {x:self.truth_params_dict[x][0] for x in self.x_names}
        elif self.truth_type == "fixed_uniform":
            self.mu = {x:self.truth_params_dict[x][0] + self.prng.uniform(self.truth_params_dict[x][1],self.truth_params_dict[x][2]) for x in self.x_names}
        elif self.truth_type == "prior_uniform":
            self.mu = {x:self.truth_params_dict[x][0] + self.prng.uniform(self.truth_params_dict[x][1]*self.truth_params_dict[x][0],self.truth_params_dict[x][2]*self.truth_params_dict[x][0]) for x in self.x_names}
        else:
            self.mu = {x:self.prng.normal(self.truth_params_dict[x][0], self.truth_params_dict[x][1]) for x in self.x_names}


    # this function gives the exogenous information that is dependent on a random process
    # In our case, exogeneous information: W^(n+1) = mu_x + eps^(n+1),
    # Where eps^(n+1) is normally distributed with mean 0 and known variance (here s.d. 0.05)
    # W^(n+1)_x : reduction in A1C level
    # self.prng.normal takes two values, mu and sigma.
    def exog_info_fn(self, decision): 
        W = self.prng.normal(self.mu[decision], self.sigma_w)
        beta_W = Beta(self.sigma_w)
        return {"reduction": W, "beta_W": beta_W, "mu": self.mu[decision]}

    # this function takes in the decision and exogenous information to return\
    # the new mu_empirical and beta values corresponding to the decision.
    def transition_fn(self, decision, exog_info):
        # for x = x_n only. Other entries unchanged.
        beta = (getattr(self.state, decision))[1] + exog_info["beta_W"]
        mu_empirical = ((getattr(self.state, decision))[1]*(getattr(self.state, decision))[0] + \
        exog_info["beta_W"]*exog_info["reduction"]) / beta
        N_x = getattr(self.state, decision)[2] + 1 # count of no. times drug x was given.
        return {decision: [mu_empirical, beta, N_x]}
    
    # this function calculates W (reduction in A1C level)
    def objective_fn(self, decision, exog_info):
        mu = exog_info["mu"]
        W = exog_info["reduction"]
        return mu
        
    # this method steps the process forward by one time increment by updating the sum of the contributions, the
    # exogenous information and the state variable
    def step(self, decision):
        # build dictionary copy of self.states (which is right now a named tuple)
        #current_state = {}
        #for k in self.state_names:
        #    current_state[k] = getattr(self.state, k)
          
        # compute new mu_empirical and beta for the decision.
        exog_info = self.exog_info_fn(decision)
        exog_info.update(self.transition_fn(decision, exog_info))
        
        # update objective  (add new W to previous obj)
        # This is the sum
        self.obj += self.objective_fn(decision, exog_info)

        current_state = {key:exog_info[decision] if key == decision else getattr(self.state, key) for key in self.state_names}
        self.state = self.build_state(current_state)

        self.t_update()
        
        # re-build self.state
        #for key in current_state:
        #    if key == decision:
        #        # replace the entry corresponding to the decision with the new exog.info
        #        current_state[decision] = exog_info[decision]
        #        # rebuild the self.state tuple using the updated dictionary current_state
        #        self.state = self.build_state(current_state)
        #    else:
        #        pass

        return exog_info
    
    # Update method for time counter
    def t_update(self):
        self.t += 1
        return self.t
    
# =============================================================================
# # UNIT TEST
# if __name__ == "__main__":
#     '''
#     this is an example of creating a model, choosing the decision at random 
#     (out of 5 possible drugs), and running the model for a fixed time period (in months)
#     '''
#     file = 'MDDMparameters.xlsx'
#     S0 = pd.read_excel(file, sheet_name = 'parameters1')
#     sigmaW = pd.read_excel(file, sheet_name = 'parameters2')
#     
#     # this needs to be (vector of 5 entries, each of which is a 2-entry vec)
#     state_names = ['M', 'Sens', 'Secr', 'AGI', 'PA']
#     # for each drug: (first entry: mu_0, second entry: beta_0, third entry: number of times drug x was given)
#     init_state = {'M': [ S0.loc['M', 'mu'] , Beta(S0.loc['M', 'sigma']), 0], \
#                   'Sens': [S0.loc['Sens', 'mu'] , Beta(S0.loc['Sens', 'sigma']), 0], \
#                   'Secr': [S0.loc['Secr', 'mu'] , Beta(S0.loc['Secr', 'sigma']), 0], \
#                   'AGI': [S0.loc['AGI', 'mu'] , Beta(S0.loc['AGI', 'sigma']), 0], \
#                   'PA': [S0.loc['PA', 'mu'] , Beta(S0.loc['PA', 'sigma']), 0]}
#     # in order: Metformin, Sensitizer, Secretagoge, Alpha-glucosidase inhibitor, Peptide analog.
#     x_names = ['M', 'Sens', 'Secr', 'AGI', 'PA']
# 
#     Model = MedicalDecisionDiabetesModel(state_names, x_names, init_state)
#     # initialize sigma_w
#     Model.sigma_w = sigmaW.loc['sigma_w', 0]
#     
#     # each time step is 1 month.
#     t_stop = 50
#     
#     
#     # Testing
#     for k in range(t_stop):
#         # policy: pick a random drug: Pure exploration.
#         decision = x_names[randint(0,4)] # policy goes here
#         # update model according to the decision
#         print("t = {}, \nstate = {} \ndecision = {}, \nobjective = {} \n".format(Model.t, Model.state, decision, Model.obj)) # better format state
#         Model.step(decision)
#         Model.t_update()
# 
# =============================================================================
