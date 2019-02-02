"""
Clinical Trials Policy class

Raluca Cobzaru (c) 2018
Adapted from code by Donghun Lee (c) 2018

"""
from collections import namedtuple
import numpy as np
from scipy.stats import binom
import scipy
import math
import pandas as pd
import copy
from ClinicalTrialsModel import ClinicalTrialsModel
import time

def trunc_poisson_fn(count, mean):
	"""
	returns list of truncated Poisson distribution with given mean and values count

	:param count: int - maximal value considered by the distribution
	:param mean: float - mean of Poisson distribution
	:return list(float) - vector of truncated Poisson pmfs 
	"""
	trunc_probs = []
	sum = 0.0
	for r in range(0, count):
		trunc_probs.insert(r, 1/math.factorial(r)*(mean**r)*np.exp(-mean))
		sum += trunc_probs[r]
	trunc_probs.insert(count, 1-sum)
	return trunc_probs

class ClinicalTrialsPolicy():
	"""
	Base class for decision policy
	"""

	def __init__(self, model, policy_names):
		"""
		initializes the policy
		
		:param model: the ClinicalTrialsModel that the policy is being implemented on
		:param policy_names: list(str) - list of policies
		"""
		self.model = model
		self.policy_names = policy_names
		self.Policy = namedtuple('Policy', policy_names)

	def build_policy(self, info):
		"""
		builds the policies depending on the parameters provided
		
		:param info: dict - contains all policy information
		:return: namedtuple - a policy object
		"""
		return self.Policy(*[info[k] for k in self.policy_names])
	
	def model_A_policy(self, state, info_tuple):
		"""
		implements deterministic lookahead policy based on Model A
		
		:param state: namedtuple - the state of the model at a given time
		:param info_tuple: tuple - contains the parameters needed to run the policy
		:return: a decision made based on the policy
		"""
		success_A = info_tuple[0]
		stop_A = info_tuple[1]
		sim_model = ClinicalTrialsModel(self.model.state_variables, self.model.decision_variables, self.model.initial_state, True)
		sim_model.state = copy.deepcopy(state)

		if stop_A==False:
			value_dict={}
			sol_dict,value_dict = model_A_value_fn(sim_model, 0, success_A,value_dict)
			new_decision = sol_dict['optimal_enroll']  
		else: 
			new_decision=0
		

		return new_decision
		
	def model_B_policy(self, state, info_tuple):
		"""
		implements lookahead policy based on Model B
		
		:param state: namedtuple - the state of the model at a given time
		:param info_tuple: tuple - contains the parameters needed to run the policy
		:return: a decision made based on the policy
		"""
		success_B = info_tuple[0]
		stop_B = info_tuple[1]
		sim_model = ClinicalTrialsModel(self.model.state_variables, self.model.decision_variables, self.model.initial_state, True)
		sim_model.state = copy.deepcopy(state)
		
		if stop_B==False:
			value_dict={}
			sol_dict,value_dict = model_B_value_fn(sim_model, 0, success_B,value_dict)
			new_decision = sol_dict['optimal_enroll']  
		else: 
			new_decision=0
		
		return new_decision
		
	def model_C_extension_policy(self, state, info_tuple):
		"""
		implements lookahead policy based on the extension of Model C
		
		:param state: namedtuple - the state of the model at a given time
		:param info_tuple: tuple - contains the parameters needed to run the policy
		:return: a decision made based on the policy
		"""
		success_C_extension = info_tuple[0]
		stop_C_extension = info_tuple[1]
		sim_model = ClinicalTrialsModel(self.model.state_variables, self.model.decision_variables, self.model.initial_state, True)
		sim_model.state = copy.deepcopy(state)

		if stop_C_extension==False:
			value_dict={}
			sol_dict,value_dict = model_C_extension_value_fn(sim_model, 0, success_C_extension,value_dict)
			new_decision = sol_dict['optimal_enroll']  
		else: 
			new_decision=0

		return new_decision
		
	def model_C_policy(self, state, info_tuple, time):
		"""
		implements hybrid policy for Model C using backward ADP 
		
		:param state: namedtuple - the state of the model at a given time
		:param info_tuple: tuple - contains the parameters needed to run the policy
		:param time: int - start time
		:return: a decision made based on the policy
		"""
		success_C = info_tuple[0]
		stop_C = info_tuple[1]
		sim_model = ClinicalTrialsModel(self.model.state_variables, self.model.decision_variables, self.model.initial_state, True)
		sim_model.state = copy.deepcopy(state)
		parameters = parameters_fn(sim_model)
		if stop_C == True: new_decision = 0
		else:
			vals = []
			decs = []
			for x_enroll in range(self.model.initial_state['enroll_min'], self.model.initial_state['enroll_max']+self.model.initial_state['enroll_step'], self.model.initial_state['enroll_step']):
				pseudo_state = [state.potential_pop + x_enroll, state.success, state.failure, state.l_response]
				if len(parameters[time]) < 8:
					value = func_simple(pseudo_state, parameters[time][0], parameters[time][1], parameters[time][2], parameters[time][3])
				else: 
					value = func(pseudo_state, parameters[time][0], parameters[time][1], parameters[time][2], parameters[time][3], parameters[time][4], parameters[time][5], parameters[time][6], parameters[time][7])
				cost = -(self.model.initial_state['program_cost'] + self.model.initial_state['patient_cost'] * x_enroll)
				vals.append(value + cost)
				decs.append(x_enroll)
			val_max = max(vals)
			new_decision = decs[vals.index(val_max)]
		return new_decision
	
	def run_policy(self, policy_info, policy, t):

		"""
		runs the model with a selected policy
		
		:param policy_info: dict - dictionary of policies and their associated parameters
		:param policy: str - the name of the chosen policy
		:param time: int - start time
		:return: float - calculated contribution
		"""
		time_run = time.time()
		model_copy = copy.deepcopy(self.model)
		
		while t <= model_copy.initial_state['trial_size'] and policy_info[policy][1] == False: 
			time_t = time.time()
			# build decision policy
			p = self.build_policy(policy_info)
			
			# implements sampled distribution for p_true
			p_true_samples = np.random.beta(model_copy.state.success, model_copy.state.failure, model_copy.initial_state['K'])
			p_belief = model_copy.state.success / (model_copy.state.success + model_copy.state.failure)
		
			# drug_success = 1 if successful, 0 if failure, -1 if continue trial (for all policies)
			if p_belief > model_copy.initial_state['theta_stop_high']:
				decision = {'prog_continue': 0, 'drug_success': 1}
				policy_info[policy][1] = True
			elif p_belief < model_copy.initial_state['theta_stop_low']:
				decision = {'prog_continue': 0, 'drug_success': 0}
				policy_info[policy][1] = True
			else:
				decision = {'prog_continue': 1, 'drug_success': -1}
			
			# makes enrollment decision based on chosen policy
			if policy == "model_A":
				decision['enroll'] = self.model_A_policy(model_copy.state, p.model_A)
			elif policy == "model_B":
				decision['enroll'] = self.model_B_policy(model_copy.state, p.model_B)
			elif policy == "model_C_extension":
				decision['enroll'] = self.model_C_extension_policy(model_copy.state, p.model_C_extension)
			elif policy == "model_C":
				decision['enroll'] = self.model_C_policy(model_copy.state, p.model_C, t)
			
			x = model_copy.build_decision(decision)
			print("Base Model t={}, obj={:,}, state.potential_pop={}, state.success={}, state.failure={}, x={}, elapsed time={:.2f} sec".format(t, model_copy.objective, 
																										model_copy.state.potential_pop, 
																										model_copy.state.success, 
																										model_copy.state.failure, x,time.time()-time_t))
			# steps the model forward one iteration
			model_copy.step(x)
			# updates policy info
			policy_info[policy][0] = decision['drug_success']
			# increments time
			t += 1
		print("Base Model: Stopping time t={}, obj(revenue)={:,}, rhobar={:.2f}, Elapsed time={:.2f} sec".format(t, model_copy.objective, model_copy.state.success/(model_copy.state.success+model_copy.state.failure),time.time()-time_run))
		policy_value = model_copy.objective
		return policy_value
		
def model_A_value_fn(model, iteration, success_index,value_dict):
	"""
	solves the deterministic shortest path problem for Model A (over given horizon);
	returns the value of the current state and its optimal number of new potential patients to enroll
	
	:param model: ClinicalTrialsModel - model which contains all state variables (physical and belief states)
	:param iteration: int - tracks the horizon in the deteministic shortest path problem
	:param success_index: int - 1 if drug is declared successful, 0 if failure, -1 if continue trial
	:return: value of current node and its optimal enrollment count
	"""
	# computes value and optimal enrollments corresponding to current state
	if success_index == -1:
		if iteration < model.initial_state['H']:
			bellman_vals = []
			bellman_decisions = []
			for x_enroll in range(model.initial_state['enroll_min'], model.initial_state['enroll_max']+model.initial_state['enroll_step'], model.initial_state['enroll_step']):

				bellman_potential_pop = model.state.potential_pop + x_enroll
				#print("Starting bellman - ite: {}, R: {}, x_enroll: {}, R_t+1: {}".format(iteration, model.state.potential_pop,x_enroll,bellman_potential_pop))
				bellman_cost = -(model.initial_state['program_cost'] + model.initial_state['patient_cost'] * x_enroll)
				bellman_state = copy.deepcopy(model.initial_state)
				bellman_state['potential_pop'] = bellman_potential_pop
				bellman_M = ClinicalTrialsModel(model.state_variables, model.decision_variables, bellman_state, True)

				value_key=(iteration+1,bellman_state['potential_pop'])
				count=-1

				# the drug success probability stays fixed
				bellman_p_belief = bellman_M.state.success / (bellman_M.state.success + bellman_M.state.failure)
				if bellman_p_belief > bellman_M.initial_state['theta_stop_high']: 
					success_index = 1
					step_value = model.initial_state['success_rev']
				elif bellman_p_belief < bellman_M.initial_state['theta_stop_low']: 
					success_index = 0
					step_value = 0
				else: 
						if value_key in value_dict:
							step_value = value_dict[value_key][0]
							count = value_dict[value_key][1]
							#print("key: {} value: {:.2f} count: {} lendict:{}".format(value_key,step_value,count,len(value_dict)))

						else:
							sol_dict,value_dict = model_A_value_fn(bellman_M, iteration+1, success_index,value_dict)
							step_value = sol_dict['value']

				value_dict.update({value_key:[step_value,count+1]})

				bellman_cost += step_value
				bellman_decisions.append(x_enroll)
				bellman_vals.append(bellman_cost)

				#print("Ending - ite: {}, R: {}, x_enroll: {}, R_t+1: {}, Cost: {}".format(iteration, model.state.potential_pop,x_enroll,bellman_potential_pop,bellman_cost))

			value = max(bellman_vals)
			optimal_enroll = bellman_decisions[bellman_vals.index(value)]
			#print("********Ending State- ite: {}, R: {}, arg_max: {}, opt_value {} ".format(iteration, model.state.potential_pop,optimal_enroll,value))
			
			return {"value": value,
				"optimal_enroll": optimal_enroll},value_dict
		# stops iterating at horizon t' = t + H
		else: 
			return {"value": 0,"optimal_enroll": 0},value_dict
	# stops experiment at node if drug is declared success or failure
	else: return {"value": model.initial_state['success_rev'] * success_index,
					"optimal_enroll": 0},value_dict
			



def model_B_value_fn(model, iteration, success_index,value_dict):
	"""
	solves the stochastic lookahead problem for Model B (over given horizon);
	returns the value of the current state and its optimal number of new potential patients to enroll
	
	:param model: ClinicalTrialsModel - model which contains all state variables (physical and belief states)
	:param iteration: int - tracks the horizon in the stochastic lookahead problem
	:param success_index: int - 1 if drug is declared successful, 0 if failure, -1 if continue trial
	:return: value of current node and its optimal enrollment count
	"""
	# computes value and optimal enrollments corresponding to current state
	if success_index == -1:
		if iteration < model.initial_state['H']:
			bellman_vals = []
			bellman_decisions = []
			for x_enroll in range(model.initial_state['enroll_min'], model.initial_state['enroll_max']+model.initial_state['enroll_step'], model.initial_state['enroll_step']):
				# "simulated" exogenous info that helps us get from (t, t') to (t, t'+1)
				bellman_potential_pop = model.state.potential_pop + x_enroll
				bellman_enrollments = math.floor(model.state.l_response * bellman_potential_pop)
				bellman_cost = -(model.initial_state['program_cost'] + model.initial_state['patient_cost'] * x_enroll)
				# loops over success values in increments of step_succ
				step_succ = int(bellman_enrollments / 3) + 1
				for set_succ in range(0, bellman_enrollments, step_succ):
					bellman_state = copy.deepcopy(model.initial_state)
					bellman_state['potential_pop'] = bellman_potential_pop
					bellman_state['success'] =  model.state.success + set_succ
					bellman_state['failure'] = model.state.failure + (bellman_enrollments - set_succ)
					bellman_M = ClinicalTrialsModel(model.state_variables, model.decision_variables, bellman_state, True)
					
					value_key=(iteration+1,bellman_state['potential_pop'],bellman_state['success'],bellman_state['failure']) #Remember to include here bellman_state['l_response'] when solving question 6
					count=-1
					
					# implements sampled distribution for bellman_p_true
					bellman_p_samples = np.random.beta(bellman_M.state.success, bellman_M.state.failure, bellman_M.initial_state['K'])
					bellman_p_belief = bellman_M.state.success / (bellman_M.state.success + bellman_M.state.failure)

					
					
					if bellman_p_belief > bellman_M.initial_state['theta_stop_high']: 
						success_index = 1
						step_value = model.initial_state['success_rev']
						#print("LA State: {}, ({}, {}), {} - Stopping time {}".format(bellman_state['potential_pop'],bellman_state['success'],bellman_state['failure'],model.state.l_response,iteration))
					elif bellman_p_belief < bellman_M.initial_state['theta_stop_low']: 
						success_index = 0
						step_value = 0
					else: 
						if value_key in value_dict:
							step_value = value_dict[value_key][0]
							count = value_dict[value_key][1]
							#print("key: {} value: {:.2f} count: {} lendict:{}".format(value_key,step_value,count,len(value_dict)))

						else:
							sol_dict,value_dict = model_B_value_fn(bellman_M, iteration+1, success_index,value_dict)
							step_value = sol_dict['value']


					
					value_dict.update({value_key:[step_value,count+1]})
					
					for k in range(0, bellman_M.initial_state['K']):
						bellman_cost += binom.pmf(set_succ, bellman_enrollments, bellman_p_samples[k]) * 1/bellman_M.initial_state['K'] * step_value
				bellman_decisions.append(x_enroll)
				bellman_vals.append(bellman_cost)
			
			value = max(bellman_vals)
			optimal_enroll = bellman_decisions[bellman_vals.index(value)]
			return {"value": value, "optimal_enroll": optimal_enroll},value_dict
		# stops iterating at horizon t' = t + H
		else: return {"value": 0,"optimal_enroll": 0},value_dict
	# stops experiment at node if drug is declared success or failure
	else: return {"value": model.initial_state['success_rev'] * success_index,"optimal_enroll": 0},value_dict



#Copy model_B_value_fn here and do the modifications for question 6
def model_C_extension_value_fn(model, iteration, success_index,value_dict):
	return {} #Get rid of this
	


# 	
def func_simple(pseudo_state, a, b, c, d):
	"""
	linear fit function for the Bellman value at given pseudo-state (for small number of data points)
	
	:param pseudo_state: list(float) - list of the four state variables for a given state
	:param a, b, c, d, e: float - parameters of the linear fit function
	"""
	sum = a*pseudo_state[0] + b*pseudo_state[1] + c*pseudo_state[2] + d*pseudo_state[3]
	return sum

def func(pseudo_state, a1, a2, b1, b2, c1, c2, d1, d2):
	"""
	quadratic fit function for the Bellman value at given pseudo-state
	
	:param pseudo_state: list(float) - list of the four state variables for a given state
	:param a1, a2, ... d2: float - parameters of the quadratic fit function
	"""
	sum = a1*pseudo_state[0]**2 + a2*pseudo_state[0]
	sum += b1*pseudo_state[1]**2 + b2*pseudo_state[1] 
	sum += c1*pseudo_state[2]**2 + c2*pseudo_state[2]  
	sum += d1*pseudo_state[3]**2 + d2*pseudo_state[3] 
	return sum
	
def parameters_fn(model):
	"""
	simulates enrollment paths; then fits the values at each state using a linear or a quadratic fit function
	returns parameters for the linear/quadratic fit functions at each t
	
	:param model: ClinicalTrialsModel - model that we simulate on (contains all state variables)
	:return parameters: list(list(float)) - parameters for the fit functions at each t
	"""
	samples = [[] for n in range(model.initial_state['sampling_size'])]
	values = [[] for n in range(model.initial_state['sampling_size'])]
	for n in range(model.initial_state['sampling_size']):
		sample_t = 0
		stop = False
		sample_M = ClinicalTrialsModel(model.state_variables, model.decision_variables, model.initial_state, True)
		while sample_t <= model.initial_state['trial_size'] and stop == False:
			p_belief = sample_M.state.success / (sample_M.state.success + sample_M.state.failure)
			p_true_samples = np.random.beta(sample_M.state.success, sample_M.state.failure, sample_M.initial_state['K'])
			# drug_success = 1 if successful, 0 if failure, -1 if continue trial
			if p_belief > model.initial_state['theta_stop_high']:
				decision = {'prog_continue': 0, 'drug_success': 1}
				stop = True
			elif p_belief < model.initial_state['theta_stop_low']:
				decision = {'prog_continue': 0, 'drug_success': 0}
				stop = True
			else:
				decision = {'prog_continue': 1, 'drug_success': -1}
			decision['enroll'] = np.random.choice(range(model.initial_state['enroll_min'], model.initial_state['enroll_max']+model.initial_state['enroll_step'], model.initial_state['enroll_step'])) if stop == False else 0
			x = sample_M.build_decision(decision)
			pseudo_state = [sample_M.state.potential_pop + decision['enroll'], sample_M.state.success, sample_M.state.failure, sample_M.state.l_response]
			sample_M.step(x)
			sample_t += 1
			samples[n].append(pseudo_state)
			values[n].append(sample_M.objective)
	parameters = []
	for t_fct in range(model.initial_state['trial_size']+1):
		samples_list = []
		values_list = []
		for n in range(model.initial_state['sampling_size']):
			if (len(samples[n]) >= (t_fct + 1)):
				samples_list.append(samples[n][t_fct])
				values_list.append(values[n][t_fct])
		samples_array = np.array(samples_list)
		values_array = np.array(values_list)
		if t_fct == 0:
			all_matrix = np.c_[samples_array, values_array]
			new_array = [tuple(row) for row in all_matrix]
			uniques = np.unique(new_array, axis=0)
			samples_array = uniques[:, 0:4]
			values_array = uniques[:, 4]
			parameters.append(np.matrix.tolist(scipy.optimize.curve_fit(func_simple, np.matrix.transpose(samples_array), values_array)[0]))
		else:
			if len(values_array) >= 8:
				parameters.append(np.matrix.tolist(scipy.optimize.curve_fit(func, np.matrix.transpose(samples_array), values_array)[0]))
			else:
				parameters.append(np.matrix.tolist(scipy.optimize.curve_fit(func_simple, np.matrix.transpose(samples_array), values_array)[0]))
	return parameters

if __name__ == "__main__":
	# this is an example of creating a model, using a chosen policy, and running until the drug is declared a success/failure or 
	# we reach the maximum number of trials
	policy_names = ['model_A', 'model_B', 'model_C', 'model_C_extension']
	state_variables = ['potential_pop', 'success', 'failure', 'l_response']
	# extracts data from given data set; defines initial state
	file = 'Trials Parameters.xlsx'
	raw_data = pd.ExcelFile(file)
	data = raw_data.parse('Exogenous Data')
	initial_state = {'potential_pop': float(data.iat[0, 0]),
					 'success': data.iat[1, 0],
					  'failure': float(data.iat[2, 0]),
					  'l_response': float(data.iat[3, 0]),
					  'theta_stop_low': data.iat[4, 0],
					  'theta_stop_high': data.iat[5, 0],
					  'alpha': data.iat[6, 0],
					  'K': int(data.iat[7, 0]),
					  'N': int(data.iat[8, 0]),
					  'trial_size': int(data.iat[9, 0]),
					  'patient_cost': data.iat[10, 0],
					  'program_cost': data.iat[11, 0],
					  'success_rev': data.iat[12, 0],
					  'sampling_size': int(data.iat[13, 0]),
					  'enroll_min': int(data.iat[14, 0]),
					  'enroll_max': int(data.iat[15, 0]),
					  'enroll_step': int(data.iat[16, 0]),
					  'H': int(data.iat[17, 0]),
					  'true_l_response': data.iat[18, 0],
					  'true_succ_rate': data.iat[19, 0]}
	decision_variables = ['enroll', 'prog_continue', 'drug_success']

	M = ClinicalTrialsModel(state_variables, decision_variables, initial_state, False)
	P = ClinicalTrialsPolicy(M, policy_names)
	t = 0
	stop = False
	policy_info = {'model_A': [-1, stop],
					'model_B': [-1, stop],
					'model_C': [-1, stop],
					'model_C_extension': [-1, stop]}
					
	# an example of running the Model B policy
	P.run_policy(policy_info, "model_B", t)
	
	pass