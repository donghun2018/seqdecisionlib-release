"""
Adaptive Market Planning Driver Script

"""
		
from collections import namedtuple
from AdaptiveMarketPlanningModel import AdaptiveMarketPlanningModel
from AdaptiveMarketPlanningPolicy import AdaptiveMarketPlanningPolicy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
	# this is an example of creating a model and running a simulation for a certain trial size

	# define state variables
	state_names = ['order_quantity', 'counter']
	init_state = {'order_quantity': 0, 'counter': 0}
	decision_names = ['step_size']
	
	# read in variables from excel file
	file = 'Base parameters.xlsx'
	raw_data = pd.ExcelFile(file)
	data = raw_data.parse('parameters')
	cost = data.iat[0, 2]
	trial_size = np.rint(data.iat[1, 2]).astype(int)
	price = data.iat[2, 2]
	theta_step = data.iat[3, 2]
	T = data.iat[4, 2]
	reward_type = data.iat[5, 2]
	
	# initialize model and store ordered quantities in an array
	M = AdaptiveMarketPlanningModel(state_names, decision_names, init_state, T,reward_type, price, cost)
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

	optimal_order_quantity = -np.log(cost/price) * 100
	print("Optimal order_quantity for price {} and cost {} is {}".format(price,cost,optimal_order_quantity))
	print("Reward type: {}, theta_step: {}, T: {} - Average reward over {} iteratios is: {}".format(reward_type,theta_step,T,trial_size,rewards_per_iteration_cum_avg[-1]))
	
	ite = np.random.randint(0,trial_size)
	order_quantity = learning_list_per_iteration[ite]
	print("Order quantity for iteration {}".format(ite))
	print(order_quantity)

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
	
	


	# ploting the analytical sol
	plt.xlabel("Time")
	plt.ylabel("Order quantity")
	plt.title("Analytical vs learned ordered quantity - (iteration {})".format(ite))
	time = np.arange(0, len(order_quantity))
	plt.plot(time, time * 0 - np.log(cost/price) * 100, label = "Analytical solution")
	plt.plot(time, order_quantity, label = "Kesten's Rule for theta_step {}".format(theta_step))
	plt.legend()
	plt.show()


	