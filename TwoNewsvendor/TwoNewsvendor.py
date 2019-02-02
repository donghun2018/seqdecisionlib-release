"""

This program implements the basic model for the two newsvendor problem. 
This code does not belong to the driverscript 


"""
from collections import namedtuple

import numpy as np
import pandas as pd
import math
import xlrd


class Exogenous_Information():

    def __init__(self, params):
        self.init_args = params
        self.prng = np.random.RandomState(params['seed'])
        self.n=0
        self.demand = None
        self.estimate_field = None
        self.estimate_central = None

    

    def generate_New_Round(self):
        self.n = self.n + 1
        self.demand = int(round(self.prng.uniform(self.init_args['dem_lower_bound'], self.init_args['dem_upper_bound'])))

        self.estimate_field = max(0,int(round(self.demand + self.prng.normal(loc = self.init_args['est_bias_field'], scale = self.init_args['est_std_field']))))
        self.estimate_central = max(0,int(round(self.demand + self.prng.normal(loc = self.init_args['est_bias_central'], scale = self.init_args['est_std_central']))))


    def get_Estimate_Field(self):
        return self.estimate_field

    def get_Estimate_Central(self):
        return self.estimate_central

    def get_Demand(self):
        return self.demand

    def get_Round_Number(self):
        return self.n



         



class Model_Field():
    """
    Base class for model
    """

    def __init__(self, state_names, x_names, s_0, params):
        """
        Initializes the model

        :param state_names: list(str) - state variable dimension names
        :param x_names: list(str) - decision variable dimension names
        :param s_0: dict - contains the information needed to populate the state names 
        with the initial state values 
        :params params: other initial information such as unit costs 
        for overage or underage and the smoothing constants 
        """

        self.init_args = params
        self.prng = np.random.RandomState(params['seed'])
        self.init_state = s_0
        self.state_names = state_names
        self.x_names = x_names
        self.State = namedtuple('State', state_names)
        self.Decision = namedtuple('Decision', x_names)
        self.pen_incurred = 0
        
        self.state = self.build_state(self.init_state)
        self.decision = None
        self.n=0
        self.exog_info = {}



    def resetModel(self,theta):
        
        self.state = self.build_state(self.init_state)
        self.decision = None
        self.n=0
        self.exog_info = {}
    
        

    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        self.decision = self.Decision(*[info[k] for k in self.x_names])
        return self.decision

    def exog_info_fn(self, decision_central, demand):
        exog_info = []
        exog_info.append(decision_central) 
        exog_info.append(demand) 
        return exog_info

    def get_alpha_bias(self):
        return self.init_args['alpha_bias']

    def updateState(self,estimate):
        state_dict = self.state._asdict()
        state_dict['estimate']=estimate
        self.state = self.build_state(state_dict)
        




    def transition_fn(self, exog_info):

        self.n +=1

        state_dict = self.state._asdict()
        
        source_bias = self.state.estimate - exog_info['demand']
        central_bias = exog_info['allocated_quantity'] - self.decision.quantity_requested

        for state_desc in ['central_bias','source_bias']:
            state_dict[state_desc] =  (1 - self.get_alpha_bias()) *  state_dict[state_desc] +  self.get_alpha_bias() * eval(state_desc)

        self.state = self.build_state(state_dict)
        

    def objective_fn(self, exog_info):
        allocated = exog_info['allocated_quantity']
        demand = exog_info['demand']
        self.pen_incurred = (self.init_args['o_field'] * max(allocated - demand, 0) + 
                  self.init_args['u_field'] * max(demand - allocated, 0))
        return -self.pen_incurred

    def showState(self,state_desc):
        return getattr(self.state,state_desc)


class Model_Central():
    """
    Base class for model
    """

    def __init__(self, state_names, x_names, s_0, params):
        """
        Initializes the model

        :param state_names: list(str) - state variable dimension names
        :param x_names: list(str) - decision variable dimension names
        :param s_0: dict - contains the information needed to populate the state names 
        with the initial state values and other initial information such as unit costs 
        for overage or underage and the smoothing constants 
        :param seed: int - seed for random number generator
        """

        self.init_args = params
        self.prng = np.random.RandomState(self.init_args['seed'])
        self.init_state = s_0
        self.state_names = state_names
        self.x_names = x_names
        self.State = namedtuple('State', state_names)
        self.Decision = namedtuple('Decision', x_names)
        self.pen_incurred =0

        self.state = self.build_state(self.init_state)
        self.decision = None
        self.n=0
        self.beta_field = 0
        self.beta_source = 0
        self.delta_field = 0
        self.delta_source = 0
        self.lambda_field = 0
        self.lambda_source = 0

    def resetModel(self,theta):
       
        self.state = self.build_state(self.init_state)
        self.decision = None
        self.n=0
        self.beta_field = 0
        self.beta_source = 0
        self.delta_field = 0
        self.delta_source = 0
        self.lambda_field = 0
        self.lambda_source = 0

        

    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        self.decision = self.Decision(*[info[k] for k in self.x_names])
        return self.decision

    def exog_info_fn(self, req_quantity, demand):
        return demand

    def updateState(self,field_request,estimate):
        state_dict = self.state._asdict()
        state_dict['field_request']=field_request
        state_dict['estimate']=estimate
        self.state = self.build_state(state_dict)

    def get_alpha_bias(self):
        return self.init_args['alpha_bias']

    def get_alpha_learning(self):
        return self.init_args['alpha_learning']



    def transition_fn(self, exog_info):

        self.n +=1

        state_dict = self.state._asdict()
        

        field_bias = self.state.field_request - exog_info['demand']
        source_bias = self.state.estimate - exog_info['demand']
        

        self.beta_field = (1 - self.get_alpha_learning()) *  self.beta_field +  self.get_alpha_learning() * (field_bias - state_dict['field_bias'])
        self.beta_source = (1 - self.get_alpha_learning()) *  self.beta_source +  self.get_alpha_learning() * (source_bias - state_dict['source_bias'])

        self.delta_field = (1 - self.get_alpha_learning()) *  self.delta_field +  self.get_alpha_learning() * ((field_bias - state_dict['field_bias'])**2)
        self.delta_source = (1 - self.get_alpha_learning()) *  self.delta_source +  self.get_alpha_learning() * ((source_bias - state_dict['source_bias'])**2)

        self.var_field = (self.delta_field-(self.beta_field**2))/(1-self.lambda_field)
        self.var_source = (self.delta_source-(self.beta_source**2))/(1-self.lambda_source)

        dem_field = self.var_field + (self.beta_field)**2
        dem_source = self.var_source + (self.beta_field)**2

        if dem_field < 0.001:
            field_w = 1
            source_w = 0
        elif dem_source < 0.001:
            field_w = 0
            source_w = 1
        else:
            field_w = 1/dem_field
            source_w = 1/dem_source
        
        sum_w = field_w + source_w

        state_dict['field_weight'] = field_w/sum_w
        state_dict['source_weight'] = source_w/sum_w

        state_dict['field_bias_hat'] = field_bias


        if self.n > 1:
            self.lambda_field = ((1 - self.get_alpha_bias())**2)*self.lambda_field + self.get_alpha_bias()**2
            self.lambda_source = ((1 - self.get_alpha_bias())**2)*self.lambda_source + self.get_alpha_bias()**2
        else:
            self.lambda_field = self.get_alpha_bias()
            self.lambda_source = self.get_alpha_bias()

        for state_desc in ['field_bias','source_bias']:
            state_dict[state_desc] =  (1 - self.get_alpha_bias()) *  state_dict[state_desc] +  self.get_alpha_bias() * eval(state_desc)

        self.state = self.build_state(state_dict)
        


    def objective_fn(self,  exog_info):
        allocated = exog_info['allocated_quantity']
        demand = exog_info['demand']
        self.pen_incurred  = (self.init_args['o_central'] * max(allocated - demand, 0) + 
                  self.init_args['u_central'] * max(demand - allocated, 0))
        return -self.pen_incurred 

    def showState(self,state_desc):
        return getattr(self.state,state_desc)

    


