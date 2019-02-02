"""
Asset selling policy class

"""
from collections import namedtuple
import pandas as pd
import numpy as np
from AssetSellingModel import AssetSellingModel
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
###
from copy import copy, deepcopy
###
import math

class AssetSellingPolicy():
    """
    Base class for decision policy
    """

    def __init__(self, model, policy_names):
        """
        Initializes the policy

        :param model: the AssetSellingModel that the policy is being implemented on
        :param policy_names: list(str) - list of policies
        """
        self.model = model
        self.policy_names = policy_names
        self.Policy = namedtuple('Policy', policy_names)

    def build_policy(self, info):
        """
        this function builds the policies depending on the parameters provided

        :param info: dict - contains all policy information
        :return: namedtuple - a policy object
        """
        return self.Policy(*[info[k] for k in self.policy_names])

    def sell_low_policy(self, state, info_tuple):
        """
        this function implements the sell-low policy

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        lower_limit = info_tuple[0]
        new_decision = {'sell': 1, 'hold': 0} if state.price < lower_limit else {'sell': 0, 'hold': 1}
        return new_decision

    def high_low_policy(self, state, info_tuple):
        """
        this function implements the high-low policy

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        lower_limit = info_tuple[0]
        upper_limit = info_tuple[1]
        new_decision = {'sell': 1, 'hold': 0} if state.price < lower_limit or state.price > upper_limit \
            else {'sell': 0, 'hold': 1}
        return new_decision

    def track_policy(self, state, info_tuple):
        """
        this function implements the track policy

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        #####
        #theta = info_tuple[0] - (1-info_tuple[1])
        theta = info_tuple[0]
        #theta = info_tuple[0]*info_tuple[1]

        prev_price = info_tuple[2]
        prev_price2 = info_tuple[3]
        
        smoothed_price_p = .7*state.price + .2*prev_price + .1*prev_price2
        smoothed_price = smoothed_price_p #discount factor 

        #state.price_d = state.price
        state_price_d = state.price * info_tuple[1]


        print('Theta {}, discount factor {},  Current price {}, smoothed_price {}, d_smoothed_prince {}, and hold interval ({}, {})'.format(theta,info_tuple[1],state_price_d,smoothed_price_p,smoothed_price,max(0,smoothed_price - theta),smoothed_price +theta))
        
        new_decision = {'sell': 1, 'hold': 0} \
            if state_price_d >= smoothed_price + theta\
                or state_price_d <= max(0,smoothed_price - theta) \
            else {'sell': 0, 'hold': 1}
        return new_decision
        #####
    def run_policy(self, param_list, policy_info, policy, time):
        """
        this function runs the model with a selected policy

        :param param_list: list of policy parameters in tuple form (read in from an Excel spreadsheet)
        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param time: float - start time
        :return: float - calculated contribution
        """
        model_copy = copy(self.model)
        theta = param_list[2][0]

        while model_copy.state.resource != 0 and time < model_copy.initial_args['T']:
            # build decision policy
            
           
            p = self.build_policy(policy_info)

            # make decision based on chosen policy
            if policy == "sell_low":
                decision = self.sell_low_policy(model_copy.state, p.sell_low)
            elif policy == "high_low":
                decision = self.high_low_policy(model_copy.state, p.high_low)
            elif policy == "track":
                decision = {'sell': 0, 'hold': 1} if time < 2 else self.track_policy(model_copy.state, p.track)

            if (time == model_copy.initial_args['T'] - 1):
                 decision = {'sell': 1, 'hold': 0}  

            x = model_copy.build_decision(decision)
            print("time={}, obj={}, s.resource={}, s.price={}, x={}".format(time, model_copy.objective,
                                                                            model_copy.state.resource,
                                                                            model_copy.state.price, x))
            #####
            prev_price2 = model_copy.state.prev_price
            #####
            
            # update previous price
            prev_price = model_copy.state.price
            
            # step the model forward one iteration
            model_copy.step(x)
            # update track policy info with new previous price
            
            #####
            policy_info.update({'track': (theta,model_copy.initial_args['gamma'] ** time) + (prev_price, prev_price2)})
            #policy_info.update({'track': param_list[2] + (prev_price, prev_price2)})
            #####
            
            # increment time
            time += 1

        contribution = model_copy.objective*model_copy.initial_args['gamma'] ** (time-1)  
        #contribution = model_copy.objective 
        print("obj={}, state.resource={}".format(contribution, model_copy.state.resource))
        
        return contribution,time


        


    def grid_search_theta_values(self, low_min, low_max, high_min, high_max, increment_size):
        """
        this function gives a list of theta values needed to run a full grid search

        :param low_min: the minimum value/lower bound of theta_low
        :param low_max: the maximum value/upper bound of theta_low
        :param high_min: the minimum value/lower bound of theta_high
        :param high_max: the maximum value/upper bound of theta_high
        :param increment_size: the increment size over the range of theta values
        :return: list - list of theta values
        """

        theta_low_values = np.linspace(low_min, low_max, (low_max - low_min) / increment_size + 1)
        theta_high_values = np.linspace(high_min, high_max, (high_max - high_min) / increment_size + 1)

        theta_values = []
        for x in theta_low_values:
            for y in theta_high_values:
                theta = (x, y)
                theta_values.append(theta)

        return theta_values, theta_low_values, theta_high_values

    def vary_theta(self, param_list, policy_info, policy, time, theta_values):
        """
        this function calculates the contribution for each theta value in a list

        :param param_list: list of policy parameters in tuple form (read in from an Excel spreadsheet)
        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param time: float - start time
        :param theta_values: list - list of all possible thetas to be tested
        :return: list - list of contribution values corresponding to each theta
        """
        contribution_values = []
       

        for theta in theta_values:
            t = time
            policy_dict = policy_info.copy()
            policy_dict.update({'high_low': theta})
            print("policy_dict={}".format(policy_dict))
            contribution = self.run_policy(param_list, policy_dict, policy, t)
            contribution_values.append(contribution)
            
        return (contribution_values)

    def plot_heat_map(self, contribution_values, theta_low_values, theta_high_values):
        """
        this function plots a heat map

        :param contribution_values: list - list of contribution values
        :param theta_low_values: list - list of theta_low_values
        :param theta_high_values: list - list of theta_high_values
        :return: none (plots a heat map)
        """
        contributions = np.array(contribution_values)
        increment_count = len(theta_low_values)
        contributions = np.reshape(contributions, (-1, increment_count))

        fig, ax = plt.subplots()
        im = ax.imshow(contributions, cmap='hot')
        # create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        # we want to show all ticks...
        ax.set_xticks(np.arange(len(theta_low_values)))
        ax.set_yticks(np.arange(len(theta_high_values)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(theta_low_values)
        ax.set_yticklabels(theta_high_values)
        # rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_title("Heatmap of contribution values across different values of theta")
        fig.tight_layout()
        plt.show()
        return True


    def plot_heat_map_many(self, contribution_values, theta_low_values, theta_high_values,iterations):
        """
        this function plots a heat map

        :param contribution_values: list - list of contribution values
        :param theta_low_values: list - list of theta_low_values
        :param theta_high_values: list - list of theta_high_values
        :return: none (plots a heat map)
        """
        fig, axsubs = plt.subplots(math.ceil(len(iterations)/2), 2)
        fig.suptitle("Heatmap of contribution values across different values of theta", fontsize=10)

        for ite,n in zip(iterations,list(range(len(iterations)))):
            contributions = np.array(contribution_values[ite])
            
            
            increment_count = len(theta_high_values)
            contributions = np.reshape(contributions, (-1, increment_count))
            contributions=contributions[::-1]
            


            print("Ite {}, n {} and plot ({},{})".format(ite,n,n // 2,n % 2))
            if (math.ceil(len(iterations)/2)>1):
                ax = axsubs[n // 2,n % 2]
            else:
                ax = axsubs[n % 2]
            
            im = ax.imshow(contributions, cmap='hot')
            cbar = ax.figure.colorbar(im, ax=ax)
            ax.set_yticks(np.arange(len(theta_low_values)))
            ax.set_xticks(np.arange(len(theta_high_values)))
            ax.set_yticklabels(list(reversed(theta_low_values)))
            ax.set_xticklabels(theta_high_values)
            
            
            # get the current labels 
            labelsx = [item.get_text() for item in ax.get_xticklabels()]
            ax.set_xticklabels([str(round(float(label), 2)) for label in labelsx])

            # get the current labels 
            labelsy = [item.get_text() for item in ax.get_yticklabels()]
            ax.set_yticklabels([str(round(float(label), 2)) for label in labelsy])


            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
            
            ax.set_title("Iteration {}".format(ite))

        # Create a big subplot
        ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        ax.set_ylabel('Theta sell low values', labelpad=0) # Use argument `labelpad` to move label downwards.
        ax.set_xlabel('Theta sell high values', labelpad=10)

        

            
        fig.tight_layout()
        plt.show()
        return True

