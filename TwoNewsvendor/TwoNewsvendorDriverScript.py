"""
Two Newsvendor as a Learning Problem 

Author: Andrei Graur

This program simulates the game with the model implemented in 
TwoNewsvendorLearning.py

Run the code with the python command, no arguments given. 

"""

import numpy as np
import pandas as pd
import math
from collections import namedtuple,defaultdict
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage


from TwoNewsvendor import Model_Field,Model_Central,Exogenous_Information
from TwoNewsvendorLearning import Learning_model_field,Learning_model_central,Choice 
from TwoNewsvendorPolicy import create_theta_grid,run_policies,printTuple, plot_heat_map
from TwoNewsvendorPolicy import Policy_Field,Policy_Central


def Main():

	t_algo_init = time.time()

	#Reading the algorithm pars
	file = 'Parameters.xlsx'
	seed = 189654913
	parDf = pd.read_excel(file, sheet_name = 'Parameters')
	parDict=parDf.set_index('Index').T.to_dict('list')
	params = {key:v for key, value in parDict.items() for v in value}
	params['seed'] = seed

	prng = np.random.RandomState(seed)
	idx = pd.IndexSlice

	print("Parameters ",params)


	#Initializing the state/decision variables for both field and central
	state_names_field = ['estimate', 'source_bias','central_bias']
	decision_names_field = ['quantity_requested','bias_applied']
	state_field = {'estimate': None, 'source_bias': 0,  'central_bias': 0}
	

	state_names_central = ['field_request', 'field_bias','field_weight','field_bias_hat','estimate','source_bias', 'source_weight']
	decision_names_central = ['quantity_allocated','bias_applied']
	state_central = {'field_request': None, 'field_bias': 0,'field_weight': 0.5, 'field_bias_hat':0,'estimate':None,'source_bias':0, 'source_weight':0.5}
	


	#Create grid for theta=(theta_field,theta_central)

	theta_grid,theta_field_list,theta_central_list = create_theta_grid(params)

	
	#Dict to hold for each theta the avg_cum_cost along the budget rounds
	avg_cum_cost_field = defaultdict(list)
	avg_cum_cost_central = defaultdict(list)
	avg_cum_cost_company = defaultdict(list)


	avg_total_cost_field = {}
	avg_total_cost_central = {}
	avg_total_cost_company = {}



	output_path = []


	for theta_field,theta_central in theta_grid:

		print("\nStarting!!!!!  policy_field: {} with par_field: {}  - policy_central: {} with par_central {}".format(params['policy_field'],theta_field,params['policy_central'],theta_central))

		if "learning" in params['policy_field']:
			M_field = Learning_model_field(theta_field, state_names_field, decision_names_field, state_field,params)
		else:
			M_field = Model_Field(state_names_field, decision_names_field, state_field,params)

		P_field = Policy_Field(params,theta_field)
	

		if "learning" in params['policy_central']:
			M_central = Learning_model_central(theta_central, state_names_central, decision_names_central, state_central,params)
		else:
			M_central = Model_Central(state_names_central, decision_names_central, state_central,params)
		
		P_central= Policy_Central(params,theta_central)

		accum_cost_field = np.zeros(params['N'])
		accum_cost_central = np.zeros(params['N'])
		accum_cost_company = np.zeros(params['N'])

		avg_request_field = 0
		avg_allocated_central = 0

		#Create Exogenous information object
		exog_info_gen = Exogenous_Information(params)

		for ite in range(params['n_Testing_Ite']):
			#print("Starting iteration {} for theta ({},{})".format(ite,theta_field,theta_central))
			t_init_ite = time.time()
			M_field.resetModel(theta_field)
			M_central.resetModel(theta_central)

			
			#Run the two_newsvendor algorithm for n=params['N'] rounds
			cost_ite_field,cost_ite_central,output_path,request_ite_field,allocated_ite_central = run_policies(ite,output_path,params,exog_info_gen,theta_field,theta_central,M_field,P_field,M_central,P_central)


			avg_request_field += request_ite_field
			avg_allocated_central += allocated_ite_central

			
			accum_cost_field = np.add(accum_cost_field,cost_ite_field)
			accum_cost_central = np.add(accum_cost_central,cost_ite_central)
			accum_cost_company = np.add(accum_cost_field,accum_cost_central)

			#print("Finishing iteration {} for theta ({},{}) in {:.2f} secs - avg costs (field,central,company):  ({:.2f}, {:.2f}, {:.2f})".format(ite,theta_field,theta_central,(time.time()-t_init_ite),cost_ite_field[-1],cost_ite_central[-1],(cost_ite_field[-1]+cost_ite_central[-1])))

			if False:
				print("Final Field State {}".format(printTuple(M_field.state)))
				print("Final Central State {}".format(printTuple(M_central.state)))

				if "learning" in params['policy_field']:
					print("Field final learning model")
					M_field.getMainParametersDf()
				if "learning" in params['policy_central']:
					print("Central final learning model")
					M_central.getMainParametersDf()
			
			
		
		avg_cum_cost_field[(theta_field,theta_central)]=list(np.divide(accum_cost_field,params['n_Testing_Ite']))
		avg_cum_cost_central[(theta_field,theta_central)]=list(np.divide(accum_cost_central,params['n_Testing_Ite']))
		avg_cum_cost_company[(theta_field,theta_central)]=list(np.divide(accum_cost_company,params['n_Testing_Ite']))


		avg_total_cost_field[(theta_field,theta_central)]=avg_cum_cost_field[(theta_field,theta_central)][-1]
		avg_total_cost_central[(theta_field,theta_central)]=avg_cum_cost_central[(theta_field,theta_central)][-1]
		avg_total_cost_company[(theta_field,theta_central)]=avg_cum_cost_company[(theta_field,theta_central)][-1]

		print("Finishing iterations for theta ({},{}). Avg reward for field: {:.2f}, avg reward for central {:.2f} and avg reward for company {:.2f}".format(theta_field,theta_central,avg_cum_cost_field[(theta_field,theta_central)][-1],avg_cum_cost_central[(theta_field,theta_central)][-1],avg_cum_cost_field[(theta_field,theta_central)][-1]+avg_cum_cost_central[(theta_field,theta_central)][-1]))

		print("Average request from field {:.2f} - Average allocated quantity by central {:.2f}".format(avg_request_field/params['n_Testing_Ite'],avg_allocated_central/params['n_Testing_Ite']))

		




	# =============================================================================
	#	 Preparing dataframe for printing and plots
	# =============================================================================
	labelsOutputPath = ['policy_field','policy_central','theta_field_theta_central','ite','n','Round','exog_est_field', 'exog_est_central','exog_demand']
	labelsOutputPath += ['field_state_' + s for s in state_names_field]
	labelsOutputPath += ['field_dec_' + s for s in decision_names_field]
	labelsOutputPath += ['central_state_' + s for s in state_names_central]
	labelsOutputPath += ['central_dec_' + s for s in decision_names_central]
	labelsOutputPath += ['util_field','accum_util_field','util_central','accum_util_central','util_company','accum_util_company']

	if "learning" in params['policy_field']:
		labelsOutputPath += M_field.getMainParametersHeaderList()
	if "learning" in params['policy_central']:
		labelsOutputPath += M_central.getMainParametersHeaderList()

	dfOutputPath = pd.DataFrame.from_records(output_path,columns=labelsOutputPath)


	dfTest= dfOutputPath.pivot_table(['field_dec_' + s for s in decision_names_field]+['central_dec_' + s for s in decision_names_central]+['central_state_field_weight','central_state_source_weight','exog_est_field', 'exog_est_central','exog_demand'],index=['theta_field_theta_central','n'])
	

	#Preparation to plot along iterations
	field_sorted_by_value = sorted(avg_cum_cost_field.items(), key=lambda kv: kv[1][-1],reverse=True)
	central_sorted_by_value = sorted(avg_cum_cost_central.items(), key=lambda kv: kv[1][-1],reverse=True)
	company_sorted_by_value = sorted(avg_cum_cost_company.items(), key=lambda kv: kv[1][-1],reverse=True)

	selected_field_positions = [i for i,dict_entry_field in enumerate(field_sorted_by_value) if dict_entry_field[0] in [central_sorted_by_value[0][0],company_sorted_by_value[0][0],central_sorted_by_value[-1][0],company_sorted_by_value[-1][0] ]]
	p=min(2,len(np.arange(1,len(field_sorted_by_value)-1)))
	displaySet = [0] + list(prng.choice(np.arange(1,len(field_sorted_by_value)-1), p, replace=False)) + [len(field_sorted_by_value)-1]

	Rounds = list(range(params['N']))

	#Preparation to plot the added bias
	dfBias= dfOutputPath.pivot_table(['field_dec_bias_applied','central_dec_bias_applied'],index=['theta_field_theta_central','n'],columns=['ite'])
	dfBias_Var = dfBias.loc[idx[:,0:params['N']],idx['field_dec_bias_applied',:]].std(level='theta_field_theta_central')

	
	maxVarIndex = dfBias_Var.max(axis=1).idxmax()
	nlargest = min(5,params['n_Testing_Ite'])
	order = np.argsort(-dfBias_Var.values, axis=1)[:, :nlargest]
	#order = prng.choice(params['n_Testing_Ite'], min(5,params['n_Testing_Ite']), replace=False)
	#order = np.ones((len(dfBias_Var),nlargest))*order

	print("\nFinishing calculations in {:.2f} secs".format(time.time()-t_algo_init))

	

	# =============================================================================
	#	 Outputing to Excel
	# =============================================================================
	if params['print_records']:

		print_init_time = time.time()

		# Create a Pandas Excel writer using XlsxWriter as the engine.
		writer = pd.ExcelWriter('DetailedOutput_{}_{}.xlsx'.format(params['policy_field'],params['policy_central']), engine='xlsxwriter')
		 # Convert the dataframe to an XlsxWriter Excel object.
		dfOutputPath.to_excel(writer, sheet_name='output')
		writer.save()  
		print_end_time = time.time() 
		print("Finished printing the excel file. Elapsed time {}\n".format(print_end_time-print_init_time))

	
	# =============================================================================
	#Ploting the results
	# =============================================================================

	#Plotting heatmaps for total cost
	figHeat,axHeat = plt.subplots(1,3,sharey=True,figsize=(16,8))
	plot_heat_map(axHeat[0],avg_total_cost_field, params,theta_field_list,theta_central_list,"Field",field_sorted_by_value)
	plot_heat_map(axHeat[1],avg_total_cost_central, params,theta_field_list,theta_central_list,"Central",central_sorted_by_value)
	plot_heat_map(axHeat[2],avg_total_cost_company, params,theta_field_list,theta_central_list,"Company",company_sorted_by_value)
	plt.suptitle("Averaged total rewards after {} time periods ({} samples) \n policy_field: {} - policy_central: {}".format(params['N'],params['n_Testing_Ite'],params['policy_field'],params['policy_central']))
	

	#Plotting the bias 
	grid_index = 0
	figBias,axBias = plt.subplots(2,1,figsize=(16,8),sharex = True)
	figBias.suptitle('Bias applied by field and central along time periods for 5 different sample paths \n policy_field: {} - policy_central: {}'.format(params['policy_field'],params['policy_central']) )

	X = [str(i) for i in Rounds[0:30]]

	#for grid_index in [dfBias_Var.index.get_loc(maxVarIndex)]:
	regular_regular = [22,len(theta_grid)-11] #(0,0) and (-10,-11)
	learning_IE_learning_IE = [0,len(theta_grid)-1]
	learning_IE_learning_IE_two_estimates = [0,len(theta_grid)-1]
	learning_IE_punishing = [0,11] #(0,0) and (-10,-11)


	i = -1
	for grid_index in eval("{}_{}".format(params['policy_field'],params['policy_central'])):
		i+=1
		theta="{}_{}".format(theta_grid[grid_index][0],theta_grid[grid_index][1])
		
		testnpField = dfBias.loc[idx[theta,0:params['N']],idx['field_dec_bias_applied',list(order[grid_index,:])]].values
		testnpCentral = dfBias.loc[idx[theta,0:params['N']],idx['central_dec_bias_applied',list(order[grid_index,:])]].values
		
		df = pd.DataFrame(testnpField, index=X)
		df.plot.bar(ax=axBias[i],legend=False)
		df=pd.DataFrame(testnpCentral, index=X)
		df.plot.bar(ax=axBias[i],legend=False,grid=True,alpha=0.7)
		axBias[i].axhline(0, 0, 1, linewidth=2, color='k')
		axBias[i].set_title(" Field_Central parameter pair ({},{})".format(theta_grid[grid_index][0],theta_grid[grid_index][1]),fontsize=10)
		axBias[1].set_xlabel('Time period')

		trans = axBias[i].get_yaxis_transform() # x in data untis, y in axes fraction
		ann = axBias[i].annotate('Field', xy=(1.02, 5), xycoords=trans)
		ann = axBias[i].annotate('Central', xy=(1.02, -5), xycoords=trans)


	#Plotting the histogram of decisions over ALL sample paths
	figHist,axHist = plt.subplots(1,2,figsize=(16,8))
	figHist.suptitle('Histogram of the bias applied by field and central along all sample paths \n policy_field: {} - policy_central: {}'.format(params['policy_field'],params['policy_central']) )

	range_list = params['bias_interval_field'].split(",")
	range_list = [int(e) for e in range_list]
	rangeF=np.arange(range_list[0],range_list[1]+2)


	range_list = params['bias_interval_central'].split(",")
	range_list = [int(e) for e in range_list]
	rangeC=np.arange(range_list[0],range_list[1]+2)

	rangeCF = np.concatenate((rangeC,rangeF))
	rangeCF, i = np.unique(rangeCF, return_index=True)

	
	i = -1
	for grid_index in eval("{}_{}".format(params['policy_field'],params['policy_central'])):
		i+=1
		
		theta="{}_{}".format(theta_grid[grid_index][0],theta_grid[grid_index][1])
			
		arrayBiasField = dfBias.loc[idx[theta,10:params['N']],idx['field_dec_bias_applied',:]].values.reshape(1,-1)
		histF = np.histogram(arrayBiasField,rangeF)

		arrayBiasCentral = dfBias.loc[idx[theta,10:params['N']],idx['central_dec_bias_applied',:]].values.reshape(1,-1)
		histC = np.histogram(arrayBiasCentral,rangeC)
		

		axHist[i].bar(histF[1][0:len(rangeF)-1],histF[0]/sum(histF[0]),label="Field")
		axHist[i].bar(histC[1][0:len(rangeC)-1],histC[0]/sum(histC[0]),label="Central")
		axHist[i].legend()
		axHist[i].set_xlabel('Bias Applied')
		axHist[i].set_title(" Field_Central parameter pair ({},{})".format(theta_grid[grid_index][0],theta_grid[grid_index][1]))
		axHist[i].set_xticks(rangeCF)
		axHist[i].set_xticklabels(rangeCF)


	#
	
	figPath, axsubs = plt.subplots(2,3,figsize=(16,8),sharex=True)
	figPath.suptitle('Avg Rewards and quantities along rounds \n policy_field: {} - policy_central: {}'.format(params['policy_field'],params['policy_central']) )
	


	#Ploting the exogenous info - last subplot
	theta_field = field_sorted_by_value[0][0][0]
	theta_central = field_sorted_by_value[0][0][1]
	p = axsubs[1,2].plot(Rounds, dfTest.loc["{}_{}".format(theta_field,theta_central),:]['exog_demand'],linestyle='dotted',label = "Avg Demand")
	axsubs[1,2].plot(Rounds, dfTest.loc["{}_{}".format(theta_field,theta_central),:]['exog_est_field'],color=p[-1].get_color(),linestyle='dashed',label = "Avg Source Estimate to Field")
	axsubs[1,2].plot(Rounds, dfTest.loc["{}_{}".format(theta_field,theta_central),:]['exog_est_central'],color=p[-1].get_color(),label = "Avg Source Estimate to Central")
	axsubs[1,2].legend(fontsize=8)
	axsubs[1,2].set_title('Exogenous Information',fontsize=10)
	axsubs[1,2].set_xlabel('Time period')
	#axsubs[1,2].set_ylabel('Units')


	for dict_entry,rank in zip(field_sorted_by_value,range(len(field_sorted_by_value))):
		theta_field,theta_central = dict_entry[0][0],dict_entry[0][1]

		rank_central = [i for i,dict_entry_central in enumerate(central_sorted_by_value) if dict_entry_central[0] == dict_entry[0] ]
		rank_company = [i for i,dict_entry_company in enumerate(company_sorted_by_value) if dict_entry_company[0] == dict_entry[0] ]


		c_f = avg_cum_cost_field[(theta_field,theta_central)][-1]
		c_c = avg_cum_cost_central[(theta_field,theta_central)][-1]
		c_t = c_f + c_c


		print("({:.2f},{:.2f}) -  Agents rewards ({:.2f},{:.2f}) and company reward {:.2f} - rank_field {}, rank_central {}, rank_company {}".format(theta_field,theta_central,c_f,c_c,c_t,rank,rank_central[0],rank_company[0]))
		theta = "{}_{}".format(theta_field,theta_central)
		if ('two_estimates' in params['policy_central'] and theta == maxVarIndex):
			plt.figure(2)
			figWeight, axWeight = plt.subplots()
			axWeight.plot(dfTest.loc[theta,:]['central_state_field_weight'],label="Weight to estimate from field")
			axWeight.plot(dfTest.loc[theta,:]['central_state_source_weight'],label="Weight to estimate from source")
			axWeight.legend()

			axWeight.set_title(r"Central weights when combining two sources of information for $(\theta^{q},\theta^{q'})=$" + "({},{})".format(theta_field,theta_central))
			axWeight.set_xlabel('Time period')

			
		
		if rank in displaySet or rank in selected_field_positions:
			#Field cost
			axsubs[0,0].plot(Rounds, np.array(avg_cum_cost_field[(theta_field,theta_central)]),linestyle='dashed',label = "({:.2f},{:.2f} - {})".format(theta_field,theta_central,rank))
			
			#Central cost
			axsubs[0,1].plot(Rounds, np.array(avg_cum_cost_central[(theta_field,theta_central)]),label = "({:.2f},{:.2f} - {})".format(theta_field,theta_central,rank_central[0]))
			
			#Company cost
			axsubs[0,2].plot(Rounds, np.array(avg_cum_cost_company[(theta_field,theta_central)]),label = "({:.2f},{:.2f} - {})".format(theta_field,theta_central,rank_company[0]))
			
			#Field - quantity requested
			axsubs[1,0].plot(Rounds, dfTest.loc["{}_{}".format(theta_field,theta_central),:]['field_dec_quantity_requested'],linestyle='dashed',label = "({},{} - {})".format(theta_field,theta_central,rank))
			
			#Central - quantity allocated
			axsubs[1,1].plot(Rounds, dfTest.loc["{}_{}".format(theta_field,theta_central),:]['central_dec_quantity_allocated'],label = "({},{} - {})".format(theta_field,theta_central,rank_central[0]))
			

		axsubs[0,0].set_title('Field Agent q - Reward',fontsize=10)
		#axsubs[0,0].set_xlabel('Time period')
		axsubs[0,0].set_ylabel('$')

		axsubs[0,1].set_title('Central Agent q\' - Reward',fontsize=10)	
		#axsubs[0,1].set_xlabel('Time period')
		#axsubs[0,1].set_ylabel('$')

		axsubs[0,2].set_title('Company q+q\' - Reward',fontsize=10)		
		#axsubs[0,2].set_xlabel('Time period')
		#axsubs[0,2].set_ylabel('$')

		axsubs[1,0].set_title('Field Agent - Avg Requested Quantity',fontsize=10)
		axsubs[1,0].set_xlabel('Time period')
		axsubs[1,0].set_ylabel('Units')

		axsubs[1,1].set_title('Central Agent - Avg Allocated Quantity',fontsize=10)
		axsubs[1,1].set_xlabel('Time period')
		#axsubs[1,1].set_ylabel('Units')
			


		axsubs[0,0].legend(title=r"$(par^q,par^{q'} - Rank^q)$",fontsize=8)
		axsubs[0,1].legend(title=r"$(par^q,par^{q'} - Rank^{q'})$",fontsize=8)
		axsubs[0,2].legend(title=r"$(par^q,par^{q'}) - Rank^{q+q'})$",fontsize=8)

	plt.show()
	plt.close('all')

if __name__ == "__main__":
    mem = max(memory_usage(proc=Main))

    print("Maximum memory used: {0} MiB".format(str(mem)))

	

