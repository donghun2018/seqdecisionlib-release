"""
Energy storage policy class

"""
from collections import namedtuple
import pandas as pd
import numpy as np
from EnergyStorageModel import EnergyStorageModel as ESM
import matplotlib.pyplot as plt
from copy import copy
import time

class EnergyStoragePolicy():
    """
    Base class for decision policy
    """

    def __init__(self, model, policy_names):
        """
        Initializes the policy

        :param model: EnergyStorageModel - the model that the policy is being implemented on
        :param policy_names: list(str) - list of policies
        """

        self.model = model
        self.policy_names = policy_names
        self.Policy = namedtuple('Policy', policy_names)

    

    def buy_low_sell_high_policy(self, time,state, theta):
        """
        this function implements the buy low, sell high policy for the ESM

        :param state: namedtuple - the state of the model at a given time
        :param theta: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        lower_limit = theta[0]
        upper_limit = theta[1]
        if state.price <= lower_limit:
            new_decision = self.model.possible_decisions[0]
        elif state.price >= upper_limit:
            new_decision = self.model.possible_decisions[1]
        else:
            new_decision = self.model.possible_decisions[2]
        return new_decision


    def bellman_policy(self, time,state, bellman_model):
        
        price = state.price
        energy = state.energy_amount
        
        maxValue = -np.inf
        maxDec = None
        for d in self.model.possible_decisions:
            x = self.model.build_decision(d, energy)
            contribution = price * (x.sell - x.buy)
           
            sum_w = 0
            w_index = 0
            for w in bellman_model.discrete_price_changes:
                f = bellman_model.f_p[w_index] if w_index == 0 else bellman_model.f_p[w_index] - bellman_model.f_p[w_index - 1]
                next_state = bellman_model.state_transition(state, x, w)
                next_v = bellman_model.values_dict[time+1][next_state] if time < bellman_model.time \
                    else bellman_model.terminal_contribution
                sum_w += f * next_v
        
                w_index += 1
            # print("w_index={}".format(w_index))
            v = contribution + sum_w
            if (v>maxValue):
                maxValue=v
                maxDec=d
        return maxDec

    def run_policy(self, policy_info, policy, stop_time):
        """
        this function runs the model with a selected policy

        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param stop_time: float - stop time
        :return: float - calculated contribution
        """
        time = 0
        model_copy = copy(self.model)
        nTrades = {'buy':0,'sell':0,'hold':0}
        buy_list = []
        sell_list = []

        while time != model_copy.init_args['T']:
            

            decision = getattr(self,policy)(time,model_copy.state, policy_info)

            #Last time period - we are going to sell energy
            if time ==  model_copy.init_args['T']-1:
                decision = {'buy': 0, 'hold': 0, 'sell': 1}   

            x = model_copy.build_decision(decision,model_copy.state.energy_amount)
            
            nTrades['buy'] += x.buy
            nTrades['sell'] += x.sell
            nTrades['hold'] += model_copy.state.energy_amount
            if x.buy>0:
                buy_list.append((time,model_copy.state.price))
            elif x.sell>0:
                sell_list.append((time,model_copy.state.price))
           
            #print("time={}, obj={}, state.energy_amount={}, state.price={}, x={}".format(time, model_copy.objective,model_copy.state.energy_amount, model_copy.state.price, x))
            
            # step the model forward one iteration
            model_copy.step(time, x)
            # increment time
            time += 1
        contribution = model_copy.objective
        
        print("Energy traded - Sell: {:.2f} - Buy: {:.2f} - Hold % : {:.2f}".format(nTrades['sell'],nTrades['buy'],nTrades['hold']/model_copy.init_args['T']))
        print("Sell times and prices ")
        for i in range(len(sell_list)):
            print("t = {:.2f} and price = {:.2f}".format(sell_list[i][0],sell_list[i][1]))
        print("Buy times and prices ")
        for i in range(len(buy_list)):
            print("t = {:.2f} and price = {:.2f}".format(buy_list[i][0],buy_list[i][1]))
        
        return contribution

    def perform_grid_search(self, params, theta_values):
        """
        this function calculates the contribution for each theta value in a list

        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param stop_time: float - stop time
        :param theta_values: list - list of all possible thetas to be tested
        :return: list - list of contribution values corresponding to each theta
        """

        tS=time.time()
        contribution_values_dict = {} 
        
        bestTheta = None
        bestContribution = - np.inf 
        
        for theta in theta_values:
            #print("Starting theta {}".format(theta))
            if theta[0] >= theta[1]:
                contribution_values_dict[theta] = 0
            else:
                contribution = self.run_policy( theta, "buy_low_sell_high_policy", params['T'])
                contribution_values_dict[theta] = contribution
                best_theta = max(contribution_values_dict, key=contribution_values_dict.get)
                print("Finishing theta {} with contribution {:.2f}. Best theta so far {}. Best contribution {:.2f}".format(theta,contribution,best_theta,contribution_values_dict[best_theta]))


        print("Finishing GridSearch in {:.2f} secs".format(time.time()-tS))
        return contribution_values_dict

    def grid_search_theta_values(self, params):
        """
        this function gives a list of theta values needed to run a full grid search

        """
        theta_buy_values = np.arange(params['theta_buy_min'],params['theta_buy_max'],params['theta_inc'])
        theta_sell_values = np.arange(params['theta_sell_min'],params['theta_sell_max'],params['theta_inc'])

        theta_values = [(x,y) for x in theta_buy_values for y in theta_sell_values]
        
        return theta_values, theta_buy_values, theta_sell_values


    def plot_heat_map(self, contribution_dict, theta_buy_values, theta_sell_values):
        """
        this function plots a heat map

        :param contribution_dict:  dict of contribution values
        :param theta_buy_values: list - list of theta_buy_values
        :param theta_sell_values: list - list of theta_sell_values
        :return: none (plots a heat map)
        """

        contribution_values = [contribution_dict[(theta_buy,theta_sell)]  for theta_sell in theta_sell_values for theta_buy in theta_buy_values]
        contributions = np.array(contribution_values)
        increment_count = len(theta_buy_values)
        contributions = np.reshape(contributions, (-1, increment_count))

        fig, ax = plt.subplots()
        im = ax.imshow(contributions, cmap='hot',origin='lower',aspect='auto')
        # create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        # we want to show all ticks...
        ax.set_xticks(np.arange(0,len(theta_buy_values),5))
        ax.set_yticks(np.arange(0,len(theta_sell_values),5))
        # ... and label them with the respective list entries
        ax.set_xticklabels(theta_buy_values[::5])
        ax.set_yticklabels(theta_sell_values[::5])
        # rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        ax.set_title("Heatmap of contribution values across different values of theta")

        ax.set_ylabel('Theta sell high values') 
        ax.set_xlabel('Theta buy low  values')

        #fig.tight_layout()
        plt.show()
        return True

    

