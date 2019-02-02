"""
Backward dynamic programming class
"""
from EnergyStorageModel import EnergyStorageModel as ESM
import numpy as np
import pandas as pd
from bisect import bisect
import matplotlib.pyplot as plt
import math
import time
from collections import namedtuple,defaultdict

class BDP():
    """
    Base class to implement backward dynamic programming
    """

    def __init__(self, discrete_prices, discrete_energy, price_changes, discrete_price_changes,
                 f_p, stop_time, model):
        """
        Initializes the model

        :param discrete_prices: list - list of discretized prices
        :param discrete_energy: list - list of discretized energy amounts
        :param price_changes: list - list of price changes
        :param discrete_price_changes: list - list of discretized price changes
        :param f_p: ndarray - contains f(p) values
        :param stop_time: int - time at which loop terminates
        :param model: energy storage model

        """
        self.discrete_energy = discrete_energy
        self.discrete_prices = discrete_prices
        self.price_changes = price_changes
        self.discrete_price_changes = discrete_price_changes
        self.f_p = f_p
        self.time = stop_time - 1
        self.model = model
        self.terminal_contribution = 0
        self.values_dict = None #this will store the vfas - it will be computed by the method bellman_2D or bellman_3D

    

    def state_transition(self, state, decision, exog_info):
        """
        this function tells us what state we transition to if we are in some state and make a decision
        (restricted to states in possible_states)

        :param state: namedtuple - the state of the model at a given time
        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info
        :return: new state object
        """


        new_energy = state.energy_amount + (self.model.init_args['eta'] * decision.buy) - decision.sell
        adjusted_new_energy = math.ceil(new_energy)


        if len(state) == 2:
            new_price = state.price + exog_info
        elif len(state) == 3:
            new_price = 0.5*state.prev_price + 0.5*state.price + exog_info
        
        if new_price <= min(self.discrete_prices):
            adjusted_new_price = min(self.discrete_prices)
        elif new_price >= max(self.discrete_prices):
            adjusted_new_price = max(self.discrete_prices)
        else:
            index = bisect(self.discrete_prices, new_price)
            adjusted_new_price = self.discrete_prices[index]


        if len(state) == 2:
            new_state = self.model.build_state({'energy_amount': adjusted_new_energy, 'price': adjusted_new_price})
        
        elif len(state) == 3:
            prev_price = state.price
            if prev_price <= min(self.discrete_prices):
                adjusted_prev_price = min(self.discrete_prices)
            elif prev_price >= max(self.discrete_prices):
                adjusted_prev_price = max(self.discrete_prices)
            else:
                index = bisect(self.discrete_prices, prev_price)
                adjusted_prev_price = self.discrete_prices[index]
            
            new_state = self.model.build_state({'energy_amount': adjusted_new_energy,
                                            'price': adjusted_new_price,
                                            'prev_price': adjusted_prev_price})

        
        return new_state

    def bellman(self):
        """
        this function computes the value function using Bellman's equation for a 2D state variable

        :return: list - list of contribution values
        """

        # make list of all possible 2D states using discretized prices and discretized energy values
        
        self.possible_states = []
        if len(self.model.state_variable) == 2:
            for price in self.discrete_prices:
                for energy in self.discrete_energy:
                    state = self.model.build_state({'energy_amount': energy,'price': price})
                    self.possible_states.append(state)
        else:
            for p in self.discrete_prices:
                for prev_p in self.discrete_prices:
                    for energy in self.discrete_energy:
                        state = self.model.build_state({'energy_amount': energy,'price': p,'prev_price': prev_p})
                        self.possible_states.append(state)

        print("State dimension: {}. State space size: {}. Exogenous info size: {}".format(len(self.model.state_variable),len(self.possible_states),len(self.discrete_price_changes)))


        time = self.time
        values = defaultdict(dict)

        while time != -1:
            max_list = {}
            for state in self.possible_states:
                price = state.price
                energy = state.energy_amount
                v_list = []
                for d in self.model.possible_decisions:
                    x = self.model.build_decision(d, energy)
                    contribution = price * (self.model.init_args['eta']*x.sell - x.buy)
                    sum_w = 0
                    w_index = 0
                    for w in self.discrete_price_changes:
                        f = self.f_p[w_index] if w_index == 0 else self.f_p[w_index] - self.f_p[w_index - 1]
                        next_state = self.state_transition(state, x, w)
                        next_v = values[time + 1][next_state] if time < self.time \
                            else self.terminal_contribution
                        sum_w += f * next_v
                        w_index += 1
                    
                    v = contribution + sum_w
                    v_list.append(v)

                max_value = max(v_list)
                decList=["Buy","Sell","Hold"]
                #print("Time: {} State: price={:.2f}, energy={:.2f} - Buy: {:.2f} Sell: {:.2f} Hold: {:.2f} - Max_value {:.2f} - maxDec {} ".format(time,price, energy,v_list[0],v_list[1],v_list[2],max_value,decList[v_list.index(max(v_list))]))
                max_list.update({state: max_value})
            values[time]=max_list
            time -= 1
        pass
        
        self.values_dict=values
        return values

    
