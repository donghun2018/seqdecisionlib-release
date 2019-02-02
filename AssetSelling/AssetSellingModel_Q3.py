"""
Asset selling model class
Adapted from code by Donghun Lee (c) 2018

"""
from collections import namedtuple
import numpy as np

class AssetSellingModel():
    """
    Base class for model
    """

    def __init__(self, state_variable, decision_variable, state_0, exog_0,T=10, gamma=1,exog_info_fn=None, transition_fn=None,
                 objective_fn=None, seed=20180529):
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

        self.initial_args = {'seed': seed,'T': T,'exog_params':exog_0,'gamma':gamma}
        exog_params = self.initial_args['exog_params']
        biasdf = exog_params['biasdf']
        biasdf = biasdf.cumsum(axis=1)
        self.initial_args['exog_params'].update({'biasdf':biasdf})
        #print(self.initial_args['exog_params']['biasdf'])
        #print("\n")
        #print(self.initial_args)

        



        

        self.prng = np.random.RandomState(seed)
        self.initial_state = state_0
        self.state_variable = state_variable
        self.decision_variable = decision_variable
        self.State = namedtuple('State', state_variable)
        self.state = self.build_state(state_0)
        self.Decision = namedtuple('Decision', decision_variable)
        self.objective = 0.0






    def build_state(self, info):
        """
        this function gives a state containing all the state information needed

        :param info: dict - contains all state information
        :return: namedtuple - a state object
        """
        return self.State(*[info[k] for k in self.state_variable])

    def build_decision(self, info):
        """
        this function gives a decision

        :param info: dict - contains all decision info
        :return: namedtuple - a decision object
        """
        return self.Decision(*[info[k] for k in self.decision_variable])


    def exog_info_fn(self):
        """
        this function gives the exogenous information that is dependent on a random process (in the case of the the asset
        selling model, it is the change in price)

        :return: dict - updated price
        """
        # we assume that the change in price is normally distributed with mean bias and variance 2

        exog_params = self.initial_args['exog_params']
        

        biasdf = exog_params['biasdf'].T
        biasprob = biasdf[self.state.bias]
        
        
        coin = self.prng.random_sample()
        if (coin < biasprob['Up']):
            new_bias = 'Up'
            bias = exog_params['UpStep']
        elif (coin>=biasprob['Up'] and coin<biasprob['Neutral']):
            new_bias = 'Neutral'
            bias = 0
        else:
            new_bias = 'Down'
            bias = exog_params['DownStep']
         
        

    
       
        #####
        prev_price2 = self.state.prev_price
        prev_price = self.state.price
        #####

        price_delta = self.prng.normal(bias, exog_params['Variance'])
        updated_price = self.state.price +  price_delta
        # we account for the fact that asset prices cannot be negative by setting the new price as 0 whenever the
        # random process gives us a negative price
        new_price = 0.0 if updated_price < 0.0 else updated_price

        print("coin ",coin," curr_bias ",self.state.bias," new_bias ",new_bias," price_delta ", price_delta, " new price ",new_price)

        #####
        return {"price": new_price,"bias":new_bias,
                    "prev_price":prev_price, "prev_price2":prev_price2}
        #####
    def transition_fn(self, decision, exog_info):
        """
        this function takes in the decision and exogenous information to update the state

        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info (in this asset selling model,
               the exogenous info does not factor into the transition function)
        :return: dict - updated resource
        """
        new_resource = 0 if decision.sell is 1 else self.state.resource
        return {"resource": new_resource}

    def objective_fn(self, decision, exog_info):
        """
        this function calculates the contribution, which depends on the decision and the price

        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info (in this asset selling model,
               the exogenous info does not factor into the objective function)
        :return: float - calculated contribution
        """
        sell_size = 1 if decision.sell is 1 and self.state.resource != 0 else 0
        obj_part =  self.state.price * sell_size
        return obj_part

    def step(self, decision):
        """
        this function steps the process forward by one time increment by updating the sum of the contributions, the
        exogenous information and the state variable

        :param decision: namedtuple - contains all decision info
        :return: none
        """
        exog_info = self.exog_info_fn()
        self.objective += self.objective_fn(decision, exog_info)
        exog_info.update(self.transition_fn(decision, exog_info))
        self.state = self.build_state(exog_info)

