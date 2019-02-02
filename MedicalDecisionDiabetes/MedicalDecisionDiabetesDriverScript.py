"""
Driver Script - Medical Decisions Diabetes Treatment

"""     

import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import math

import time

from MedicalDecisionDiabetesModel import MedicalDecisionDiabetesModel as MDDM
from MedicalDecisionDiabetesModel import Beta
from MedicalDecisionDiabetesPolicy import MDDMPolicy


def formatFloatList(L,p):
    sFormat = "{{:.{}f}} ".format(p) * len(L) 
    outL = sFormat.format(*L)
    return outL.split()

def normalizeCounter(counter):
    total = sum(counter.values(), 0.0)
    for key in counter:
        counter[key] /= total
    return counter

        
# unit testing
if __name__ == "__main__":
    '''
    this is an example of creating a model, choosing the decision according to the policy of choice, 
    and running the model for a fixed time period (in months)
    '''
    
    # initial parameters
    seed = 19783167
    print_excel_file=False
    


    # in order: Metformin, Sensitizer, Secretagoge, Alpha-glucosidase inhibitor, Peptide analog.
    x_names = ['M', 'Sens', 'Secr', 'AGI', 'PA']
    policy_names = ['UCB', 'IE', 'PureExploitation', 'PureExploration']

    
    #reading parameter file and initializing variables
    file = 'MDDMparameters.xlsx'
    S0 = pd.read_excel(file, sheet_name = 'parameters1')
    additional_params = pd.read_excel(file, sheet_name = 'parameters2')
    
    

    policy_str = additional_params.loc['policy', 0]
    policy_list = policy_str.split()
    

    # each time step is 1 month.
    t_stop = int(additional_params.loc['N', 0]) # number of times we test the drugs
    L = int(additional_params.loc['L', 0]) # number of samples
    theta_range_1 = np.arange(additional_params.loc['theta_start', 0],\
                              additional_params.loc['theta_end', 0],\
                              additional_params.loc['increment', 0])

    # dictionaries to store the stats for different values of theta
    theta_obj = {p:[] for p in policy_names}
    theta_obj_std = {p:[] for p in policy_names}

    #data structures to output the algorithm details
    mu_star_labels = [x+"_mu_*" for x in x_names]
    mu_labels = [x+"_mu_bar" for x in x_names]
    sigma_labels = [x+"_sigma_bar" for x in x_names]
    N_labels = [x+"_N_bar" for x in x_names]
    labelsOutputPath = ["Policy","Truth_type","Theta","Sample_path"] + mu_star_labels + ["Best_Treatment", "n"] + mu_labels + sigma_labels + N_labels + ["Decision","W","CumReward","isBest"]        
    
    output_path = []

    # data structures to accumulate best treatment count
    best_treat = {(p,theta):[] for p in policy_list for theta in theta_range_1} #one list for each (p,theta) pair - each list is along the sample paths 
    best_treat_count_list = {(p,theta):[] for p in policy_list for theta in theta_range_1} #one list for each (p,theta) pair - the list is along the sample paths - each element of the list is accumulated during the experiments
    best_treat_Counter_hist = {(p,theta):[] for p in policy_list for theta in theta_range_1 }
    best_treat_Chosen_hist = {(p,theta):[] for p in policy_list for theta in theta_range_1 }

     # data structures to accumulate the decisions
    decision_Given_Best_Treat_list = {(p,theta,d):[] for p in policy_list for theta in theta_range_1  for d in x_names}
    decision_Given_Best_Treat_Counter = {(p,theta,d):[] for p in policy_list for theta in theta_range_1  for d in x_names}
    
    decision_ALL_list = {(p,theta):[] for p in policy_list for theta in theta_range_1 }
    decision_ALL_Counter = {(p,theta):[] for p in policy_list for theta in theta_range_1 }
    
    

    #initialing the model
    Model = MDDM(x_names, x_names, S0, additional_params)
    Model.printTruth()
    Model.printState()
    

    P = MDDMPolicy(Model, policy_names,seed)
     
    
# =============================================================================
#    running the policy
# =============================================================================
    
    for policy_chosen in policy_list:
        print("Starting policy {}".format(policy_chosen)) 
        
        
        P_make_decision = getattr(P,policy_chosen)       
    # loop over theta (theta)
        policy_start = time.time()
        for theta in theta_range_1:
            Model.prng = np.random.RandomState(seed)
            P.prng = np.random.RandomState(seed)
            
            F_hat = []
            last_state_dict = {x:[0.,0.,0] for x in x_names }
            states_avg = {}

            
            # loop over sample paths
            for l in range(1,L+1):
                
                # get a fresh copy of the model
                model_copy = copy(Model)
                
                # sample the truth - the truth is going to be the same for the N experiments in the budget
                model_copy.exog_info_sample_mu()
                #print("sampled_mu: ", formatFloatList(list(model_copy.mu.values()),3) )

                
                # determine the best treatment for the sampled truth
                best_treatment = max(model_copy.mu, key=model_copy.mu.get)
                best_treat[(policy_chosen,theta)].append(best_treatment)
                best_treat_count = 0
                decision_list = []
                
                # prepare record for output
                mu_output = [model_copy.mu[x] for x in x_names]
                record_sample_l =  [policy_chosen, Model.truth_type,theta,l] + mu_output + [best_treatment] 
                
                
                
                # loop over time (N, in notes)
                for n in range(t_stop):

                    # formating pre-decision state for output
                    state_mu = [getattr(model_copy.state,x)[0] for x in x_names]
                    state_sigma = [1/math.sqrt(getattr(model_copy.state,x)[1]) for x in x_names]
                    state_N = [getattr(model_copy.state,x)[2] for x in x_names]
                    
                    
                    # make decision based on chosen policy
                    decision = P_make_decision(model_copy, theta)
                    decision_list.append(decision)
                    
                    
                    # step forward in time sampling the reduction, updating the objective fucntion and the state
                    exog_info = model_copy.step(decision)
                    best_treat_count += decision==best_treatment and 1 or 0

                
                    # adding record for output
                    record_sample_t = [n] + state_mu + state_sigma + state_N  + [decision, exog_info['reduction'],model_copy.obj,decision==best_treatment and 1 or 0]
                    output_path.append(record_sample_l + record_sample_t)
                    
                    

                # updating end of experiments stats
                F_hat.append(model_copy.obj)
                last_state_dict.update({x:[last_state_dict[x][0] + getattr(model_copy.state,x)[0],last_state_dict[x][1] + getattr(model_copy.state,x)[1],last_state_dict[x][2] + getattr(model_copy.state,x)[2]] for x in x_names })
                best_treat_count_list[(policy_chosen,theta)].append(best_treat_count)
                decision_Given_Best_Treat_list[(policy_chosen,theta,best_treatment)] += decision_list
                decision_ALL_list[(policy_chosen,theta)] += decision_list

               
            
            # updating end of theta stats
            F_hat_mean = np.array(F_hat).mean()
            F_hat_var = np.sum(np.square(np.array(F_hat) -  F_hat_mean))/(L-1)
            theta_obj[policy_chosen].append(F_hat_mean)
            theta_obj_std[policy_chosen].append(np.sqrt(F_hat_var/L))
            print("Finishing policy = {}, Truth_type {} and theta = {}. F_bar_mean = {:.3f} and F_bar_std = {:.3f}".format(policy_chosen,Model.truth_type,theta,F_hat_mean,np.sqrt(F_hat_var/L)))
            states_avg = {x:[last_state_dict[x][0]/L,last_state_dict[x][1]/L,last_state_dict[x][2]/L] for x in x_names}
            print("Averages along {} iterations and {} budget trial:".format(L,t_stop))
            for x in x_names:
                print("Treatment {}: m_bar {:.2f}, beta_bar {:.2f} and N {}".format(x,states_avg[x][0],states_avg[x][1],states_avg[x][2]))
            

            best_treat_Counter = Counter(best_treat[(policy_chosen,theta)])
            best_treat_Counter_hist.update({(policy_chosen,theta):best_treat_Counter})
            
            hist, bin_edges = np.histogram(np.array(best_treat_count_list[(policy_chosen,theta)]), t_stop)
            best_treat_Chosen_hist.update({(policy_chosen,theta):hist})

            

            print("Histogram best_treatment")
            print(normalizeCounter(best_treat_Counter))
            
            print("Histogram decisions")
            decision_ALL_Counter[(policy_chosen,theta)] = normalizeCounter(Counter(decision_ALL_list[(policy_chosen,theta)] ))
            print(decision_ALL_Counter[(policy_chosen,theta)])

            decision_Given_Best_Treat_dict = {x:dict(normalizeCounter(Counter(decision_Given_Best_Treat_list[(policy_chosen,theta,x)]))) for x in Model.x_names}
            decision_df = pd.DataFrame(decision_Given_Best_Treat_dict)
            print(decision_df.head())
            print("\n\n")

            
         
         # updating end of policy stats   
        policy_end = time.time()
        print("Ending policy {}. Elapsed time {} secs\n\n\n".format(policy_chosen,policy_end - policy_start))

# =============================================================================
#     Outputing to Excel
# =============================================================================
    if print_excel_file:
        print_init_time = time.time()
        dfOutputPath = pd.DataFrame.from_records(output_path,columns=labelsOutputPath)
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('DetailedOutput{}.xlsx'.format(Model.truth_type), engine='xlsxwriter')
         # Convert the dataframe to an XlsxWriter Excel object.
        dfOutputPath.to_excel(writer, sheet_name='output')
        writer.save()  
        print_end_time = time.time() 
        print("Finished printing the excel file. Elapsed time {}".format(print_end_time-print_init_time))

# =============================================================================
#     Generating Plots
# =============================================================================
    
    l = len(theta_range_1)
    inc = additional_params.loc['increment', 0] 


    fig1, axsubs = plt.subplots(1,2)
    fig1.suptitle('Comparison of policies for the Medical Decisions Diabetes Model: \n (N = {}, L = {}, Truth_type = {} )'.format(t_stop, L, Model.truth_type) )
    
    color_list = ['b','g','r','m']
    nPolicies = list(range(len(policy_list)))

    for policy_chosen,p in zip(policy_list,nPolicies):

        axsubs[0].plot(theta_range_1, theta_obj[policy_chosen], "{}o-".format(color_list[p]),label = "{}".format(policy_chosen))
        axsubs[0].set_title('Mean')
        axsubs[0].legend()
        axsubs[0].set_xlabel('theta')
        axsubs[0].set_ylabel('estimated value for (F_bar)')
      
        axsubs[1].plot(theta_range_1, theta_obj_std[policy_chosen], "{}+:".format(color_list[p]),label = "{}".format(policy_chosen))
        axsubs[1].set_title('Std')
        axsubs[1].legend()
        axsubs[1].set_xlabel('theta')
        #axsubs[1].set_ylabel('estimated value for (F_bar)')

    
    plt.show()
    fig1.savefig('Policy_Comparison_{}.jpg'.format(Model.truth_type))

    #fig = plt.figure()
    #plt.title('Comparison of policies for the Medical Decisions Diabetes Model: \n (N = {}, L = {}, Truth_type = {} )'.format(t_stop, L, Model.truth_type))
    #color_list = ['b','g','r','m']
    #nPolicies = list(range(len(policy_list)))
    #for policy_chosen,p in zip(policy_list,nPolicies):
    #    plt.plot(theta_range_1, theta_obj[policy_chosen], "{}o-".format(color_list[p]),label = "mean for {}".format(policy_chosen))
    #    if plot_std:
    #        plt.plot(theta_range_1, theta_obj_std[policy_chosen], "{}+:".format(color_list[p]),label = "std for {}".format(policy_chosen))
    #plt.legend()
    #plt.xlabel('theta')
    #plt.ylabel('estimated value (F_bar)')
    #plt.show()
    #fig.savefig('Policy_Comparison_{}.jpg'.format(Model.truth_type))

