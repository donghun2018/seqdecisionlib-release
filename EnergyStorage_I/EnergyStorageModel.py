"""
Energy storage model class
Adapted from code by Donghun Lee (c) 2018

"""
from collections import namedtuple
import numpy as np
import pandas as pd

class EnergyStorageModel():
    """
    Base class for energy storage model
    """

    def __init__(self, state_variable, decision_variable, state_0, params, exog_params,possible_decisions,
                 exog_info_fn=None, transition_fn=None, objective_fn=None):
        """
        Initializes the model

        :param state_variable: list(str) - state variable dimension names
        :param decision_variable: list(str) - decision variable dimension names
        :param state_0: dict - contains the information to populate initial state, including eta (the fraction of
               energy maintained when charging or discharging the battery) and battery capacity
        :param params: all the parameters including DataFrame (exog_data) containning the price information 
        :param possible_decisions: list - list of possible decisions we could make
        :param exog_info_fn: function - calculates relevant exogenous information
        :param transition_fn: function - takes in decision variables and exogenous information to describe how the state
               evolves
        :param objective_fn: function - calculates contribution at time t
        """

        self.init_args = params
        self.prng = np.random.RandomState(params['seed'])
        self.exog_params = exog_params

        self.initial_state = state_0
        self.state_variable = state_variable
        self.decision_variable = decision_variable
        
        self.possible_decisions = possible_decisions
        self.State = namedtuple('State', state_variable)
        self.state = self.build_state(self.initial_state)
        self.Decision = namedtuple('Decision', decision_variable)
        self.objective = 0.0
        
        #This will keep a list of states visited
        self.states = [self.state]

    def reset(self):
        self.objective = 0.0
        self.state = self.build_state(self.initial_state)
        self.states = [self.state]



    def build_state(self, info):
        """
        this function returns a state containing all the state information needed

        :param info: dict - contains all state information
        :return: namedtuple - a state object
        """
        return self.State(*[info[k] for k in self.state_variable])

    def build_decision(self, info,energy_amount):
        """
        this function returns a decision

        :param info: dict - contains all decision info
        :param energy_amount: float - amount of energy
        :return: namedtuple - a decision object

        """
        info_copy = {'buy': 0, 'hold': 0, 'sell': 0}
        # the amount of power that can be bought or sold is limited by constraints
        for k in self.decision_variable:
            if k == 'buy' and info[k] > 0:
                info_copy[k] = (self.init_args['Rmax'] - energy_amount) / self.init_args['eta']
            elif k == 'sell' and info[k] > energy_amount:
                info_copy[k] = energy_amount
            else:
                info_copy[k] = info[k]
        return self.Decision(*[info_copy[k] for k in self.decision_variable])

    def exog_info_fn(self,time):
        
        next_price = self.exog_params['hist_price'][time]
        
        return next_price

    def transition_fn(self, time, decision):
        """
        this function takes in the decision and exogenous information to update the state

        :param time: int - time at which the state is at
        :param decision: namedtuple - contains all decision info
        :return: updated state
        """
        new_price = self.exog_info_fn(time)
        new_energy_amount = self.state.energy_amount + (self.init_args['eta'] * decision.buy) - decision.sell

        

        if len(self.state_variable) == 2:
            state = self.build_state({'energy_amount': new_energy_amount,'price': new_price})
        
        elif len(self.state_variable) == 3:
            state = self.build_state({'energy_amount': new_energy_amount,'price': new_price,'prev_price':self.state.price})

        
        return state

    def objective_fn(self, decision):
        """
        this function calculates the contribution, which depends on the decision and the price

        :param decision: namedtuple - contains all decision info
        :return: float - calculated contribution
        """
        obj_part = self.state.price * (self.init_args['eta']*decision.sell - decision.buy)
        return obj_part

    def step(self, time, decision):
        """
        this function steps the process forward by one time increment by updating the sum of the contributions
        and the state variable

        :param time: int - time at which the state is at
        :param decision: decision: namedtuple - contains all decision info
        :return: none
        """
        self.objective += self.objective_fn(decision)
        self.state = self.transition_fn(time, decision)
        self.states.append(self.state)

