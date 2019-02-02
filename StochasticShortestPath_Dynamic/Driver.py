"""
Stochastic Shortest Paths - Learning the costs
Dynamic Model - search for the parameter theta, 
which represents the percentile of the distribution 
of each cost to use to make sure we get a penalty as
small as possible. Run it using python command.

Author: Andrei Graur 
"""

from collections import namedtuple
import math
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd

from Model import StaticModel
from GraphGenerator import GraphGenerator
from Policy import LookaheadPolicy




if __name__ == "__main__":
	
	file = 'Parameters.xlsx'
	seed = 189654913
	METRIC = "PERCENTILE"


	#Reading the algorithm pars
	parDf = pd.read_excel(file, sheet_name = 'Parameters')
	parDict=parDf.set_index('Index').T.to_dict('list')
	params = {key:v for key, value in parDict.items() for v in value}
	params['seed'] = seed
	theta_list = params['theta_cost_set'].split()

	print("Parameters ",params)

	#Initializing the network
	G = GraphGenerator(params)
	if params['networkType'] == 'Steps':
		nTries = G.createNetworkSteps()
	else:
		nTries = G.createNetworkChance()

	print("Created network in {} tries. From origin {} to destination {}. Number of steps is {} and the average cost is {:.2f}. The deadline to define lateness will be {:.2f}".format(nTries,G.start_node,G.end_node,G.steps,G.get_avg_cost_paths(),G.get_deadline()))
	#input("Press Enter to continue...")
	


	# Initializing the model
	state_names = ['node']
	init_state = {'node': G.start_node}
	decision_names = ['nextNode']

	M = StaticModel(state_names, decision_names, init_state, params, G)

	
	# Initialing the lists that will hold the results
	x = []
	avgCostList = []
	avgPenaltyList = []
	avgStepsList = []
	
	# Iterating over theta
	for theta in theta_list:
		
		theta = float(theta)
		M.start_new_theta(theta)
		x.append(theta)
		
		cost, penalty, steps = M.runTrials(params['nIterations'],G.get_deadline())

		avgCostList.append(cost)
		avgPenaltyList.append(penalty)
		avgStepsList.append(steps)

		print("Avg total cost with parameter {0} is {1:.3f}. Probability of being late is {2:.2f} and avg number of steps is {3:.2f}\n ".format(theta, cost, penalty,steps))
	
	

	print("ThetaCost ",x)
	print("AvgCost ",avgCostList)
	print("ProbLateness ",avgPenaltyList)
	print("AvgSteps ",avgStepsList)

	#Ploting the results
	fig1, axsubs = plt.subplots(1,2)
	fig1.suptitle('Comparison of theta^cost -  origin {}, destination {}, dist {} - deadline {} and number of iterations {}'.format(M.G.start_node,M.G.end_node,M.G.steps,G.get_deadline(),params['nIterations']) )
  

	axsubs[0].plot(x, avgCostList)
	axsubs[0].set_title('Average Cost')
	axsubs[0].set_xlabel('Percentile')
	axsubs[0].set_ylabel('$')

	axsubs[1].plot(x, avgPenaltyList)
	axsubs[1].set_title('Probability of being late (Risk) ')
	axsubs[1].set_xlabel('Percentile')
	axsubs[1].set_ylabel('%')

	
	plt.show()



	

	pass
