"""
Asset selling driver script
"""

from collections import namedtuple
import pandas as pd
import numpy as np
from AssetSellingModel import AssetSellingModel
from AssetSellingPolicy import AssetSellingPolicy
import matplotlib.pyplot as plt
from copy import copy
import math

if __name__ == "__main__":
    # read in policy parameters from an Excel spreadsheet, "asset_selling_policy_parameters.xlsx"
    sheet1 = pd.read_excel("asset_selling_policy_parameters.xlsx", sheet_name="Sheet1")
    params = zip(sheet1['param1'], sheet1['param2'])
    param_list = list(params)
    sheet2 = pd.read_excel("asset_selling_policy_parameters.xlsx", sheet_name="Sheet2")
    sheet3 = pd.read_excel("asset_selling_policy_parameters.xlsx", sheet_name="Sheet3")
    biasdf = pd.read_excel("asset_selling_policy_parameters.xlsx", sheet_name="Sheet4")
    
    
   
    policy_selected = sheet3['Policy'][0]
    T = sheet3['TimeHorizon'][0]
    initPrice = sheet3['InitialPrice'][0]
    initBias = sheet3['InitialBias'][0]
    
    exog_params = {'UpStep':sheet3['UpStep'][0],'DownStep':sheet3['DownStep'][0],'Variance':sheet3['Variance'][0],'biasdf':biasdf}
   
    nIterations = sheet3['Iterations'][0] 
    printStep = sheet3['PrintStep'][0]
    printIterations = [0]
    printIterations.extend(list(reversed(range(nIterations-1,0,-printStep))))  
    
    
    print("exog_params ",exog_params)
   
    # initialize the model and the policy
    policy_names = ['sell_low', 'high_low', 'track']
    state_names = ['price', 'resource','bias']
    init_state = {'price': initPrice, 'resource': 1,'bias':initBias}
    decision_names = ['sell', 'hold']

    
    M = AssetSellingModel(state_names, decision_names, init_state,exog_params,T)
    P = AssetSellingPolicy(M, policy_names)
    t = 0
    prev_price = init_state['price']


    # make a policy_info dict object
    policy_info = {'sell_low': param_list[0],
                   'high_low': param_list[1],
                   'track': param_list[2] + (prev_price,)}
    
    
    if (not policy_selected =='full_grid'):
        print("Selected policy {}, time horizon {}, initial price {} and number of iterations {}".format(policy_selected,T,initPrice,nIterations))
        contribution_iterations=[P.run_policy(param_list, policy_info, policy_selected, t) for ite in list(range(nIterations))]

        contribution_iterations = pd.Series(contribution_iterations)
        print("Contribution per iteration: ")
        print(contribution_iterations)
        cum_avg_contrib = contribution_iterations.expanding().mean()
        print("Cumulative average contribution per iteration: ")
        print(cum_avg_contrib)
        
        #plotting the results
       
        fig, axsubs = plt.subplots(1,2,sharex=True,sharey=True)
        fig.suptitle("Asset selling using policy {} with parameters {} and T {}".format(policy_selected,policy_info[policy_selected],T) )
        i = np.arange(0, nIterations, 1)
        
        axsubs[0].plot(i, cum_avg_contrib, 'g')
        axsubs[0].set_title('Cumulative average contribution')
          
        axsubs[1].plot(i, contribution_iterations, 'g')
        axsubs[1].set_title('Contribution per iteration')
        
    
        # Create a big subplot
        ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        ax.set_ylabel('USD', labelpad=0) # Use argument `labelpad` to move label downwards.
        ax.set_xlabel('Iterations', labelpad=10)
        
        plt.show()
        
    else:
        # obtain the theta values to carry out a full grid search
        grid_search_theta_values = P.grid_search_theta_values(sheet2['low_min'], sheet2['low_max'], sheet2['high_min'], sheet2['high_max'], sheet2['increment_size'])
        # use those theta values to calculate corresponding contribution values
        
        contribution_iterations = [P.vary_theta(param_list, policy_info, "high_low", t, grid_search_theta_values[0]) for ite in list(range(nIterations))]
        
        contribution_iterations_arr = np.array(contribution_iterations)
        cum_sum_contrib = contribution_iterations_arr.cumsum(axis=0)
        nElem = np.arange(1,cum_sum_contrib.shape[0]+1).reshape((cum_sum_contrib.shape[0],1))
        cum_avg_contrib=cum_sum_contrib/nElem
        print("cum_avg_contrib")
        print(cum_avg_contrib)
    
        # plot those contribution values on a heat map
        P.plot_heat_map_many(cum_avg_contrib, grid_search_theta_values[1], grid_search_theta_values[2], printIterations)
        
        