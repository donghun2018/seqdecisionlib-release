'''

The policy for the two agent newsvendor game.

'''

from TwoNewsvendor import Model_Field
from TwoNewsvendor import Model_Central

import numpy as np
import math

import matplotlib.pyplot as plt

def printTuple(a):
    printStr = ""
    for f in a._fields:
        printStr += " {}: {:.2f}".format(f,getattr(a, f))
    return printStr 

def printTupleValues(a):
    printStr = ""
    for f in a._fields:
        printStr += "{:.2f} ".format(getattr(a, f))
    return printStr


def formatFloatList(L,p):
    sFormat = "{{:.{}f}} ".format(p) * len(L) 
    outL = sFormat.format(*L)
    return outL.split()



def plot_heat_map(ax,contribution_dict, params,theta_field_values, theta_central_values,titleString,player_sorted_by_value):
        """
        this function plots a heat map

        
        """

#       
        textcolors=["black", "white"]

        contribution_values = [contribution_dict[(theta_field,theta_central)]  for theta_central in theta_central_values for theta_field in theta_field_values]
        contributions = np.array(contribution_values)
        increment_count = len(theta_field_values)
        contributions = np.reshape(contributions, (-1, increment_count))

        
        
        
        im = ax.imshow(contributions, cmap='hot',origin='lower',aspect='auto',alpha=.9)
        threshold = im.norm(contributions.max())/2
        # create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        # we want to show all ticks...
        ax.set_xticks(np.arange(len(theta_field_values)))
        ax.set_yticks(np.arange(len(theta_central_values)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(theta_field_values)
        ax.set_yticklabels(theta_central_values)
        # rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        ax.set_title(titleString)

        for rank_player,dict_entry_player in enumerate(player_sorted_by_value):

            y_ind = [i for i,y in enumerate(theta_central_values) if dict_entry_player[0][1] == y]
            x_ind = [i for i,x in enumerate(theta_field_values) if dict_entry_player[0][0] == x]

            text = ax.text(x_ind[0], y_ind[0], "{}\n {:.0f}".format(rank_player,dict_entry_player[1][-1]), ha="center", va="center", color=textcolors[im.norm(dict_entry_player[1][-1]) < threshold],fontsize=7)

            #text = ax.text(x_ind[0], y_ind[0], "{}_{}".format(dict_entry_player[0][0], dict_entry_player[0][1]), ha="center", va="center", color=textcolors[im.norm(dict_entry_player[1][-1]) < threshold],fontsize=7)

        if params['policy_central']=='regular' or params['policy_central']=='punishing':
            ax.set_ylabel(r'$bias^{central}$',fontsize=14) 
        elif 'learning' in params['policy_central']:
            ax.set_ylabel(r'$\theta^{central}$',fontsize=14) 


        if params['policy_field']=='regular':
            ax.set_xlabel(r'$bias^{field}$',fontsize=14)
        elif 'learning' in params['policy_field']:
            ax.set_xlabel(r'$\theta^{field}$',fontsize=14)
        

        

        #fig.tight_layout()
       
        return True


def create_theta_grid(params):
    
    #Field
    if params['policy_field']=='regular':
        range_list = params['bias_interval_field'].split(",")
        range_list = [int(e) for e in range_list]
        theta_field_list=list(range(range_list[0],range_list[1]+1))
    
    elif 'learning' in params['policy_field']:
        if isinstance(params['theta_set_field'], str):
            theta_field_list = params['theta_set_field'].split(",")
            theta_field_list = [float(e) for e in theta_field_list]
        else:
            theta_field_list = [float(params['theta_set_field'])] 

    #Central
    if params['policy_central']=='regular' or params['policy_central']=='punishing':
        range_list = params['bias_interval_central'].split(",")
        range_list = [int(e) for e in range_list]
        theta_central_list=list(range(range_list[0],range_list[1]+1))
    
    elif 'learning' in params['policy_central']:
        if isinstance(params['theta_set_central'], str):
            theta_central_list = params['theta_set_central'].split(",")
            theta_central_list = [float(e) for e in theta_central_list]
        else:
            theta_central_list = [float(params['theta_set_central'])] 

    theta_grid = []
    for theta_field in theta_field_list:
        for theta_central in theta_central_list:
            theta_grid.append((theta_field,theta_central))

    return theta_grid,theta_field_list,theta_central_list


def run_policies(ite,record_budget,params,exog_info_gen,theta_field,theta_central,M_field,P_field,M_central,P_central):

    cost_ite_field = []
    cost_ite_central = []

    accum_util_field = 0
    accum_util_central = 0

    accum_request_field = 0
    accum_allocated_central=0

    record_sample_ite = [params['policy_field'],params['policy_central'],"{}_{}".format(theta_field,theta_central),ite]
    
    for n in range(params['N']):
        #Generate exogenous info - estimates and demand - but we are not observing the demand
        exog_info_gen.generate_New_Round()
        #print("Round {} - Estimate for the field {}, estimate for central {} and true demand {}".format(exog_info_gen.get_Round_Number(),exog_info_gen.get_Estimate_Field(),exog_info_gen.get_Estimate_Central(),exog_info_gen.get_Demand()))
        record_sample_t = [n,exog_info_gen.get_Round_Number(),exog_info_gen.get_Estimate_Field(),exog_info_gen.get_Estimate_Central(),exog_info_gen.get_Demand()]

        #Field  updates its state variable with an estimate
        M_field.updateState(exog_info_gen.get_Estimate_Field())
        #print("Field State {}".format(printTuple(M_field.state)))
        record_sample_t += list(M_field.state)

        #Field makes a decision
        field_request,bias_field = P_field.getDecision(M_field)
        M_field.build_decision({'quantity_requested': field_request,'bias_applied':bias_field})
        accum_request_field += field_request
        #print("Field Decision {}".format(printTuple(M_field.decision)))
        record_sample_t += list(M_field.decision)

        #Central updates its state with field request and (possibly) an external estimate
        M_central.updateState(field_request,exog_info_gen.get_Estimate_Central())
        #print("Central State {}".format(printTuple(M_central.state)))
        record_sample_t += list(M_central.state)

        #Central makes a decision
        decision_central,bias_central = P_central.getDecision(M_central)
        M_central.build_decision({'quantity_allocated': decision_central,'bias_applied':bias_central})
        accum_allocated_central += decision_central
        #print("Central Decision {}".format(printTuple(M_central.decision)))
        record_sample_t += list(M_central.decision)

        #True demand is revelead
        demand = exog_info_gen.get_Demand()
        exog_info_pos_dec = {'allocated_quantity': decision_central, 'demand': demand}
        
        #Costs/penalties for field and central are computed
        util_field = M_field.objective_fn(exog_info_pos_dec)
        util_central = M_central.objective_fn(exog_info_pos_dec)
        #print("Field utility {:.2f} - Central utility {:.2f}".format(util_field,util_central))
        
        accum_util_field += util_field
        accum_util_central += util_central
        
        #record_sample_t += formatFloatList([util_field,accum_util_field,util_central,accum_util_central],2)
        util_company = util_field + util_central
        accum_util_company = accum_util_field + accum_util_central

        record_sample_t += [util_field,accum_util_field,util_central,accum_util_central,util_company,accum_util_company]


        cost_ite_field.append(accum_util_field)
        cost_ite_central.append(accum_util_central)

        

        #Field and Central transition to next round updating all the stats
        M_field.transition_fn(exog_info_pos_dec)
        M_central.transition_fn(exog_info_pos_dec)

        if "learning" in params['policy_field']:
            record_sample_t +=  M_field.getMainParametersList()
        if "learning" in params['policy_central']:
            record_sample_t +=  M_central.getMainParametersList()

        record_budget.append(record_sample_ite+record_sample_t)
        
    return cost_ite_field,cost_ite_central,record_budget,accum_request_field/params['N'],accum_allocated_central/params['N']




class Policy_Field():


    def __init__(self, params,theta):
        self.init_args = params
        self.theta = theta


    def getDecision(self,model):
        decision=getattr(self,self.init_args['policy_field'])
        return decision(model)

    def getLearningBias(self,model):
        
        if ("UCB" in model.init_args['policy_field']):
            stats = {x:model.choices[x].get_UCB_value(model.n + 1) for x in model.choice_range}
        else:
            stats = {x:model.choices[x].get_IE_value() for x in model.choice_range}

        bias = max(stats,key=stats.get)
        return bias



    def regular(self, model):
        #ATTENTION! In this policy, self.theta is the bias that  field is adding - one of the values in the parameter interval "bias_interval_field"
        decision = round(model.state.estimate - model.state.source_bias - model.state.central_bias + self.theta)
        #bias = decision - (model.state.estimate - model.state.source_bias)
        bias = self.theta
        return decision, bias


    def learning_UCB(self,model):
        bias = self.getLearningBias(model)
        decision = round(model.state.estimate - model.state.source_bias + bias)
        return decision,bias
        

        return decision,bias

    def learning_IE(self, model):
        # This method implements the Interval Estimation policy

        bias = self.getLearningBias(model)
        decision = round(model.state.estimate - model.state.source_bias  + bias)

        return decision,bias

    

class Policy_Central():

    def __init__(self, params,theta):
        self.init_args = params
        self.theta = theta


    def getDecision(self,model):
        decision=getattr(self,self.init_args['policy_central'])
        return decision(model)

    def getLearningBias(self,model):

        if ("UCB" in model.init_args['policy_central']):
            stats = {x:model.choices[x].get_UCB_value(model.n + 1) for x in model.choice_range}
        else:
            stats = {x:model.choices[x].get_IE_value() for x in model.choice_range}

        bias = max(stats,key=stats.get)
        return bias


    def regular(self, model):
        #ATTENTION! In this policy, self.theta is the bias that  central is adding - one of the values in the parameter interval "bias_interval_central"
        decision = round(model.state.field_request - model.state.field_bias  + self.theta)
        decision = max(0,decision)
        #bias = decision - model.state.field_request
        bias = self.theta
        return decision, bias

    def punishing(self, model):
        if model.state.field_bias_hat >0:
            decision = round(model.state.field_request - 2 * model.state.field_bias_hat)
            bias = - 2 * model.state.field_bias_hat
        else:
            #decision = round(model.state.field_request - model.state.field_bias  + self.theta)
            decision = round(model.state.field_request  + self.theta)
            bias = self.theta

        decision = max(0,decision)
        #bias = decision - model.state.field_request
        return decision, bias

    def learning_UCB(self,model):
        bias = self.getLearningBias(model)
        decision = round(model.state.field_request + bias)
        return max(0,decision),bias
        

    def learning_IE(self, model):
        # This method implements the Interval Estimation policy

        bias = self.getLearningBias(model)
        decision = round(model.state.field_request  + bias)
        decision = max(0,decision)
        return decision,bias

    def learning_IE_two_estimates(self, model):
        bias = self.getLearningBias(model)
        decision = round(model.state.field_weight * (model.state.field_request) + model.state.source_weight * (model.state.estimate  -  model.state.source_bias) + bias)

        return max(0,decision),bias





