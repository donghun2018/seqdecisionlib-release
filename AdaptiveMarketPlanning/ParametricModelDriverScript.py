"""
Parametric Model Driver Script

"""
	
from collections import namedtuple
from ParametricModel import ParametricModel
from AdaptiveMarketPlanningPolicy import AdaptiveMarketPlanningPolicy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
	# this is an example of creating a model and running a simulation for a certain trial size
	
	# define state variables
	state_names = ['counter', 'price', 'theta']
	init_state = {'counter': 0, 'price': 26, 'theta': np.array([1, 1, 1])}
	decision_names = ['step_size']
	
	# read in variables from excel file
	file = 'ParametricModel parameters.xlsx'
	raw_data = pd.ExcelFile(file)
	data = raw_data.parse('parameters')
	cost = data.iat[0, 2]
	trial_size = np.rint(data.iat[1, 2]).astype(int)
	price_low = data.iat[2, 2]
	price_high = data.iat[3, 2]
	theta_step = data.iat[4, 2]
	T = data.iat[5, 2]
	reward_type = data.iat[6, 2]

	# initialize model and run simulations
	M = ParametricModel(state_names, decision_names, init_state, T, reward_type,cost, price_low = price_low, price_high = price_high)
	print("Theta_step ",theta_step)
	P = AdaptiveMarketPlanningPolicy(M, theta_step)

	rewards_per_iteration = []
	learning_list_per_iteration = []
	for ite in list(range(trial_size)):
		print("Starting iteration ", ite)
		reward,learning_list = P.run_policy()
		M.learning_list=[]
		#print(learning_list)
		rewards_per_iteration.append(reward)
		learning_list_per_iteration.append(learning_list)
		print("Ending iteration ", ite," Reward ",reward)


	nElem = np.arange(1,trial_size+1)
	
	rewards_per_iteration = np.array(rewards_per_iteration)
	rewards_per_iteration_sum = rewards_per_iteration.cumsum()
	rewards_per_iteration_cum_avg = rewards_per_iteration_sum/nElem

	if (reward_type=="Cumulative"):
		rewards_per_iteration_cum_avg = rewards_per_iteration_cum_avg/T
		rewards_per_iteration = rewards_per_iteration/T

	
	print("Reward type: {}, theta_step: {}, T: {} - Average reward over {} iteratios is: {}".format(reward_type,theta_step,T,trial_size,rewards_per_iteration_cum_avg[-1]))

	price = np.arange(price_low, price_high, 1)
	optimal = -np.log(cost/price) * 100
	df = pd.DataFrame({'Price' : price, 'OptOrderQuantity' : optimal})
	print(df)
	
	ite = np.random.randint(0,trial_size)
	theta_ite = learning_list_per_iteration[ite]
	#print("Thetas for iteration {}".format(ite))
	#print(theta_ite)

	#Ploting the reward
	fig1, axsubs = plt.subplots(1,2,sharex=True,sharey=True)
	fig1.suptitle("Reward type: {}, theta_step: {}, T: {}".format(reward_type,theta_step,T) )
	
	axsubs[0].plot(nElem, rewards_per_iteration_cum_avg, 'g')
	axsubs[0].set_title('Cum_average reward')
	  
	axsubs[1].plot(nElem, rewards_per_iteration, 'g')
	axsubs[1].set_title('Reward per iteration')
	#Create a big subplot
	ax = fig1.add_subplot(111, frameon=False)
	# hide tick and tick label of the big axes
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax.set_ylabel('USD', labelpad=0) # Use argument `labelpad` to move label downwards.
	ax.set_xlabel('Iterations', labelpad=10)
	plt.show()
		
	
	if (False):
		for i in range(trial_size):
			M.step(AdaptiveMarketPlanningPolicy(M, theta_step).kesten_rule())
		
		# plot results
		price = np.arange(price_low, price_high, 0.1)
		optimal = -np.log(cost/price) * 100
		plt.plot(price, optimal, color = 'green', label = "analytical solution")
		order_quantity = [M.order_quantity_fn(k, M.state.theta) for k in price]
		plt.plot(price, order_quantity, color = 'blue', label = "parametrized solution")
		plt.legend()
		plt.show()