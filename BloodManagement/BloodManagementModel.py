import numpy as np
from collections import (namedtuple, defaultdict)

class Model():
    """
    Base class for model
    """

    def __init__(self, state_names, x_names, init_state, Bld_Net,params):
        
        self.params = params
        self.init_state = init_state
        self.state_names = state_names
        self.x_names = x_names
        self.State = namedtuple('State', state_names)
        self.state = self.build_state(init_state)
        self.Decision = namedtuple('Decision', x_names)
        self.obj = 0.0
        self.Bld_Net = Bld_Net
        self.bld_inv = init_state['BloodInventory']
        self.demand = init_state['Demand']
        self.donation = init_state['Donation']
        
        # include initial inventory into the network
        for i in range(self.params['NUM_BLD_NODES']):
            self.Bld_Net.bloodamount[i] = self.bld_inv[i]
        
        # include initial demand into the network
        for i in range(self.params['NUM_DEM_NODES']):
            self.Bld_Net.demandamount[i] = self.demand[i]

        
    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        return self.Decision(*[info[k] for k in self.x_names])
    
    # exogenous information = demand from t-1 to t and new donated blood
    def exog_info_fn(self, exog_info):
        self.demand  = exog_info.demand
        # update the demand nodes
        for i in range(self.params['NUM_DEM_NODES']):
            self.Bld_Net.demandamount[i] = exog_info.demand[i]
        # save the donation vector to the model
        self.donation = exog_info.donation
        return exog_info
    
    def transition_fn(self, decision):
        # iterate through hold vector
        hold = decision[0]
        for i in range(self.params['NUM_BLD_NODES']):
            self.Bld_Net.holdamount[i] = hold[i]
            
        
        rev_don = list(reversed(self.donation))
        rev_hld = list(reversed(self.Bld_Net.holdamount))
        # age the blood at hold node and add in the donations
        
        for i in range(self.params['NUM_BLD_NODES']):
            if (i % self.params['MAX_AGE'] == self.params['MAX_AGE']-1):
                # add donation
                rev_hld[i] = rev_don[i // self.params['MAX_AGE']]
            else:
                # age
                rev_hld[i] = rev_hld[i+1]
        
        rev_hld = list(reversed(rev_hld))
        # amount at blood node = amount at hold node
        for i in range(self.params['NUM_BLD_NODES']):
            self.Bld_Net.bloodamount[i] = rev_hld[i]  
        
        # updating obj value       
        self.obj += decision[1] 
        
        # update current state
        self.bld_inv = self.Bld_Net.bloodamount
        return self.state

    def objective_fn(self):
        return self.obj
    
########################################################################################################
    
class Exog_Info():
    def __init__(self, demand, donation):
        # list consisting of blood demand objects
        self.demand = demand
        # list consisting of blood unit objects donated to the blood inventory
        self.donation = donation


# function to generate random exogenous information dependent on blood type and time t   
def generate_exog_info_by_bloodtype(t, Bld_Net, params):
    # demand
    demand= []
    if (t in params['TIME_PERIODS_SURGE'] and np.random.uniform(0, 1) < params['SURGE_PROB']):
        factor = params['SURGE_FACTOR']
    else:
        factor = 0
    demand = [round(np.random.uniform(0, params['MAX_DEM_BY_BLOOD'][dmd[0]]*params['SURGERYTYPES_PROP'][dmd[1]]*params['SUBSTITUTION_PROP'][dmd[2]])) + factor*int(np.random.poisson(params['MAX_DEM_BY_BLOOD'][dmd[0]]*params['SURGERYTYPES_PROP'][dmd[1]]*params['SUBSTITUTION_PROP'][dmd[2]]))  for dmd in Bld_Net.demandnodes]    
    
    # donation
    donation = [round(np.random.uniform(0, params['MAX_DON_BY_BLOOD'][i])) for i in params['Bloodtypes']]    
    return Exog_Info(demand, donation)


# function to generate random exogenous information dependent on blood type and time t 
def generate_exog_info_by_bloodtype_p(t, Bld_Net, params):
    # demand
    if (t in params['TIME_PERIODS_SURGE'] and np.random.uniform(0, 1) < params['SURGE_PROB']):
        factor = params['SURGE_FACTOR']
    else:
        factor = 1
    
    demand = [int(np.random.poisson(factor*params['MAX_DEM_BY_BLOOD'][dmd[0]]*params['SURGERYTYPES_PROP'][dmd[1]]*params['SUBSTITUTION_PROP'][dmd[2]])) for dmd in Bld_Net.demandnodes]   

    if False:
        demand=[]
        for dmd in Bld_Net.demandnodes: 
            if dmd[0]=="O-":
                if dmd[1]=="Urgent":
                    demand.append(1)
                else:
                    eleDem=max(0,int(np.random.poisson(factor*params['MAX_DEM_BY_BLOOD'][dmd[0]]-1))-1)
                    demand.append(eleDem)

            else:
                demand.append(int(np.random.poisson(factor*params['MAX_DEM_BY_BLOOD'][dmd[0]]*params['SURGERYTYPES_PROP'][dmd[1]]*params['SUBSTITUTION_PROP'][dmd[2]])))  


    
    #donation
    donation = [int(np.random.poisson(params['MAX_DON_BY_BLOOD'][i])) for i in params['Bloodtypes']]    
    
    return Exog_Info(demand, donation)



##########################################################################################################
# function to calculate one step contribution 
def contribution(params,bloodnode, demandnode):
    
    # if substutition is not allowed
    if (demandnode[2] == False and bloodnode[0] != demandnode[0]) or (demandnode[2] == True and params['SubMatrix'][(bloodnode[0], demandnode[0])] == False):
        value=params['INFEASIABLE_SUBSTITUTION_PENALTY']
    else:
        # start giving a bonus depending on the age of the blood
        #value = params['AGE_BONUS'][int(bloodnode[1])]
        value=0
        # no substitution
        if bloodnode[0] == demandnode[0]:
            value += params['NO_SUBSTITUTION_BONUS']
        # filling urgent demand
        if demandnode[1] == 'Urgent':
            value += params['URGENT_DEMAND_BONUS']
        # filling elective demand
        else:
            value += params['ELECTIVE_DEMAND_BONUS']

        if demandnode[1] == 'Elective':
            value += params['BLOOD_FOR_ELECTIVE_PENALTY']
   
    

    return(value)                  