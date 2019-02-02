"""
Energy storage driver script

"""
import time
from collections import namedtuple
import pandas as pd
import numpy as np
from EnergyStorageModel import EnergyStorageModel as ESM
from EnergyStoragePolicy import EnergyStoragePolicy
from BackwardDP import BDP
import matplotlib.pyplot as plt
from copy import copy
from scipy.ndimage.interpolation import shift
import pickle
from bisect import bisect



def process_raw_price_data(file,params):
    DISC_TYPE = "FROM_CUM"
    #DISC_TYPE = "OTHER"

    print("Processing raw price data. Constructing price change list and cdf using {}".format(DISC_TYPE))
    tS = time.time()

    # load energy price data from the Excel spreadsheet
    raw_data = pd.read_excel(file, sheet_name="Raw Data")

    # look at data spanning a week
    data_selection = raw_data.iloc[0:params['T'], 0:5]

    # rename columns to remove spaces (otherwise we can't access them)
    cols = data_selection.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
    data_selection.columns = cols

    # sort prices in ascending order
    sort_by_price = data_selection.sort_values('PJM_RT_LMP')
    #print(sort_by_price.head())

    hist_price = np.array(data_selection['PJM_RT_LMP'].tolist())
    #print(hist_price[0])

    max_price = pd.DataFrame.max(sort_by_price['PJM_RT_LMP'])
    min_price = pd.DataFrame.min(sort_by_price['PJM_RT_LMP'])
    print("Min price {:.2f} and Max price {:.2f}".format(min_price,max_price))
    



    # sort prices in ascending order
    sort_by_price = data_selection.sort_values('PJM_RT_LMP')

    # calculate change in price and sort values of change in price in ascending order
    data_selection['Price_Shift'] = data_selection.PJM_RT_LMP.shift(1)
    data_selection['Price_Change'] = data_selection['PJM_RT_LMP'] - data_selection['Price_Shift']
    sort_price_change = data_selection.sort_values('Price_Change')
    

    # discretize change in price and obtain f(p) for each price change
    max_price_change = pd.DataFrame.max(sort_price_change['Price_Change'])
    min_price_change = pd.DataFrame.min(sort_price_change['Price_Change'])
    print("Min price change {:.2f} and Max price change {:.2f}".format(min_price_change,max_price_change))
    
    
    

    # there are 191 values for price change
    price_changes_sorted = sort_price_change['Price_Change'].tolist()
    # remove the last NaN value
    price_changes_sorted.pop()

    if DISC_TYPE == "FROM_CUM":
    # discretize price change  by interpolating from cumulative distribution
        xp = price_changes_sorted
        fp = np.arange(len(price_changes_sorted) - 1) / (len(price_changes_sorted) - 1)
        cum_fn = np.append(fp, 1)

        # obtain 30 discrete prices
        discrete_price_change_cdf = np.linspace(0, 1, params['nPriceChangeInc'])
        discrete_price_change_list = []
        for i in discrete_price_change_cdf:
            interpolated_point = np.interp(i, cum_fn, xp)
            discrete_price_change_list.append(interpolated_point)
    else:
        price_change_range = max_price_change - min_price_change
        price_change_increment = price_change_range / params['nPriceChangeInc']
        discrete_price_change = np.arange(min_price_change, max_price_change, price_change_increment)
        discrete_price_change_list = list(np.append(discrete_price_change, max_price_change))


        f_p = np.arange(len(price_changes_sorted) - 1) / (len(price_changes_sorted) - 1)
        cum_fn = np.append(f_p, 1)
        discrete_price_change_cdf = []
        for c in discrete_price_change_list:
            interpolated_point = np.interp(c, price_changes_sorted, cum_fn)
            discrete_price_change_cdf.append(interpolated_point)

    price_changes_sorted = np.array(price_changes_sorted)
    discrete_price_change_list = np.array(discrete_price_change_list)
    discrete_price_change_cdf = np.array(discrete_price_change_cdf)
    discrete_price_change_pdf = discrete_price_change_cdf - shift(discrete_price_change_cdf,1,cval=0)

    mean_price_change = np.dot(discrete_price_change_list,discrete_price_change_pdf)

    #print("discrete_price_change_list ",discrete_price_change_list)
    #print("discrete_price_change_cdf",discrete_price_change_cdf)
    #print("discrete_price_change_pdf",discrete_price_change_pdf)


    print("Finishing processing raw price data in {:.2f} secs. Expected price change is {:.2f}. Hist_price len is {}".format(time.time()-tS,mean_price_change,len(hist_price)))
    #input("enter any key to continue...") 

    exog_params = {'hist_price':hist_price,"price_changes_sorted":price_changes_sorted,"discrete_price_change_list":discrete_price_change_list,"discrete_price_change_cdf":discrete_price_change_cdf}

    return exog_params


if __name__ == "__main__":
    

    file = 'Parameters.xlsx'
    seed = 189654913


    #Reading the algorithm pars
    parDf = pd.read_excel(file, sheet_name = 'ParamsModel')
    parDict=parDf.set_index('Index').T.to_dict('list')
    params = {key:v for key, value in parDict.items() for v in value} 
    params['seed'] = seed
    params['T'] = min(params['T'],192)
    

    parDf = pd.read_excel(file, sheet_name = 'GridSearch')
    parDict=parDf.set_index('Index').T.to_dict('list')
    paramsPolicy = {key:v for key, value in parDict.items() for v in value}
    params.update(paramsPolicy)

    parDf = pd.read_excel(file, sheet_name = 'BackwardDP')
    parDict=parDf.set_index('Index').T.to_dict('list')
    paramsPolicy = {key:v for key, value in parDict.items() for v in value}
    params.update(paramsPolicy)

    if isinstance(params['priceDiscSet'], str):
        price_disc_list = params['priceDiscSet'].split(",")
        price_disc_list = [float(e) for e in price_disc_list]
    else:
        price_disc_list = [float(params['priceDiscSet'])] 
    params['price_disc_list']=price_disc_list

    print("Parameters ",params)
    #input("enter any key to continue...")


    #exog_params  is a dictionary with  three lists: hist_price, price_changes_list, discrete_price_change_cdf
    exog_params = process_raw_price_data(file,params)
          
    # create a model and a policy
    policy_names = ['buy_low_sell_high_policy','bellman_policy']
    state_variable = ['price', 'energy_amount']
    initial_state = {'price': exog_params['hist_price'][0],
                     'energy_amount':params['R0'] }
    decision_variable = ['buy', 'hold', 'sell']
    possible_decisions = [{'buy': 1, 'hold': 0, 'sell': 0}, {'buy': 0, 'hold': 0, 'sell': 1},
                          {'buy': 0, 'hold': 1, 'sell': 0}]
    M = ESM(state_variable, decision_variable, initial_state, params, exog_params,possible_decisions)
    P = EnergyStoragePolicy(M, policy_names)

    ##########################################################################
    #GridSearch
    if params['Algorithm']=='GridSearch':
        # obtain the theta values to carry out a full grid search
        grid_search_theta_values = P.grid_search_theta_values(params)
        print(grid_search_theta_values)
        #input("enter any key to continue...")
        
        # use those theta values to calculate corresponding contribution values
        contribution_values_dict = P.perform_grid_search(params, grid_search_theta_values[0])

        # plot those contribution values on a heat map, with theta_buy on the horizontal axis and theta_sell on the
        # vertical axis
        P.plot_heat_map(contribution_values_dict, grid_search_theta_values[1], grid_search_theta_values[2])
    ##################################################################################

    #################################################################################
    #BackwardDP
    if params['Algorithm']=='BackwardDP':
        #Constructing the state space
        # make list of possible energy amount stored at a time
        discrete_energy = np.array([0.,1.])

        # make list of prices with different increments
        min_price = np.min(exog_params['hist_price'])
        max_price = np.max(exog_params['hist_price'])
        
        
        for inc in params['price_disc_list']:
            discrete_prices = np.arange(min_price,max_price+inc,inc)

            print("\nStarting BackwardDP 2D")
            test_2D = BDP(discrete_prices, discrete_energy, exog_params['price_changes_sorted'], exog_params['discrete_price_change_list'], exog_params['discrete_price_change_cdf'], params['T'], copy(M))
        
            # 2D states - time the process with a 2D state variable
            t0 = time.time()
            value_dict = test_2D.bellman()
            t1 = time.time()
            time_elapsed = t1-t0
            print("Time_elapsed_2D_model={:.2f} secs.".format(time_elapsed))

            
            print("Starting policy evaluation for the actual sample path")
            tS=time.time()
            contribution = P.run_policy(test_2D, "bellman_policy", params['T'])
            print("Contribution using BackwardDP 2D is {:.2f}. Finished in {:.2f}s".format(contribution,time.time()-tS))


            
            if params['run3D']:
                print("\nStarting BackwardDP 3D")

                state_variable_3 = ['price', 'energy_amount','prev_price']
                
                index = bisect(discrete_prices, exog_params['hist_price'][1])
                adjusted_p1 = discrete_prices[index]
                index = bisect(discrete_prices, exog_params['hist_price'][0])
                adjusted_p0 = discrete_prices[index]
                initial_state_3 = {'price': adjusted_p1,'energy_amount':params['R0'], 'prev_price':adjusted_p0}
                
                M3 = ESM(state_variable_3, decision_variable, initial_state_3, params, exog_params,possible_decisions)
                P3 = EnergyStoragePolicy(M3, policy_names)

                test_3D = BDP(discrete_prices, discrete_energy, exog_params['price_changes_sorted'], exog_params['discrete_price_change_list'], exog_params['discrete_price_change_cdf'], params['T'], copy(M3))


                t0 = time.time()
                value_dict = test_3D.bellman()
                t1 = time.time()
                time_elapsed = t1-t0
                print("Time_elapsed_3D_model={:.2f} secs.".format(time_elapsed))

                
                
                print("Starting policy evaluation for the actual sample path")
                tS=time.time()
                contribution = P3.run_policy(test_3D, "bellman_policy", params['T'])
                print("Contribution using BackwardDP 3D is {:.2f}. Finished in {:.2f}s".format(contribution,time.time()-tS))



    #########################################################################


    