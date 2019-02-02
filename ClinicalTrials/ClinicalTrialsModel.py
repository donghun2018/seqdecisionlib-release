"""
Clinical Trials Model class

Raluca Cobzaru (c) 2018
Adapted from code by Donghun Lee (c) 2018

"""
from collections import namedtuple
import numpy as np
from scipy.stats import binom
import math
import pandas as pd

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

def mc_success_fn(count, mean, samples, N, K):
	"""
	simulates enrollment and success process using Monte Carlo sampling
	
	:param count: int - count of new potential patients
	:param mean: float - mean of the truncated Poisson distribution
	:param samples: list(float) - samples for the true success rate (assuming sampled distribution)
	:param N: int - number of Monte Carlo samples
	:param K: int - number of samples for the true success rate
	:return: dict - number of enrollments and successes
	"""
	enrollment_samples = []
	success_samples = []
	trunc_probs = trunc_poisson_fn(count, mean)
	# simulates enrollment process using truncated Poisson probabilities
	for n in range(N):
		success_samples.append(0)
		MC_r_sample = np.random.choice(range(count+1), size=None, replace=True, p=trunc_probs)
		enrollment_samples.append(MC_r_sample)
		MC_prob_sample = samples[np.random.randint(0, K)]
		# simulates success count using sample probability
		for k in range(0, MC_r_sample):
			bernoulli_sim = np.random.uniform(0, 1)
			if bernoulli_sim < MC_prob_sample:
				success_samples[n] += 1
	# uniformly chooses enrollments and successes from MC samples
	enrolled = np.random.choice(enrollment_samples)
	return {"mc_enroll": enrolled, 
			"mc_success": success_samples[enrollment_samples.index(enrolled)]}

class ClinicalTrialsModel():
	"""
	Base class for model
	"""

	def __init__(self, state_variables, decision_variables, s_0, simulation, exog_info_fn=None, transition_fn=None,
				 objective_fn=None, seed=20180529):
		"""
		Initializes the model

		:param state_variables: list(str) - state variable dimension names
		:param decision_variables: list(str) - decision variable dimension names
		:param s_0: dict - needs to contain at least the information to populate initial state using state_names
		:param simulation: bool - if True, simulates exogenous data; if False, uses data from given dataset
		:param exog_info_fn: function - calculates relevant exogenous information
		:param transition_fn: function - takes in decision variables and exogenous information to describe how the state
			   evolves
		:param objective_fn: function - calculates contribution at time t
		:param seed: int - seed for random number generator
		"""

		self.init_args = {seed: seed}
		self.prng = np.random.RandomState(seed)
		self.initial_state = s_0
		self.state_variables = state_variables
		self.State = namedtuple('State', state_variables)
		self.state = self.build_state(s_0)
		self.simulation = simulation
		self.decision_variables = decision_variables
		self.Decision = namedtuple('Decision', decision_variables)
		self.objective = 0.0

	def build_state(self, info):
		"""
        returns a state containing all the given state information
		
        :param info: dict - all state information
        :return: namedtuple - a state object
        """
		return self.State(*[info[k] for k in self.state_variables])

	def build_decision(self, info):
		"""
        returns a decision containing all the given deicison information
        
		:param info: dict - all decision info
        :return: namedtuple - a decision object
        """
		return self.Decision(*[info[k] for k in self.decision_variables])



	def exog_info_fn(self, decision):
		"""
        returns the exogenous information dependent on a random process
        :param decision: int - number of new potential patients
		:return: dict - new enrollments and the number of successes among them
        """
		if self.simulation == False:
			exog_patients = math.floor(np.random.poisson(lam=self.initial_state['true_l_response'] * (self.state.potential_pop + decision.enroll), size=None))
			exog_succ = math.floor(np.random.binomial(exog_patients, self.initial_state['true_succ_rate'], size=None))


			#exog_patients = math.floor(self.initial_state['true_l_response'] * (self.state.potential_pop + decision.enroll))
			#exog_succ = math.floor(self.initial_state['true_succ_rate'] * exog_patients)

			return {"new_patients": exog_patients,
					"succ_count": exog_succ}
		else:
			r_bar = math.floor(self.state.l_response * (self.state.potential_pop + decision.enroll))
			# implements new patients and success process using Monte Carlo sampling
			p_true_samples = np.random.beta(self.state.success, self.state.failure, self.initial_state['K'])
			MC_samples = mc_success_fn(decision.enroll, r_bar, p_true_samples, self.initial_state['N'], self.initial_state['K'])
			return {"new_patients": MC_samples['mc_enroll'],
					"succ_count": MC_samples['mc_success']}
		
	
	def transition_fn(self, decision, exog_info):
		"""
        updates the state given the decision and exogenous information
		:param decision: namedtuple - contains all decision info
        :param exog_info: contains all exogenous information
        :return: dict - updated state
        """
		enroll_pop = decision.prog_continue * (self.state.potential_pop + decision.enroll)
		new_lambda = (1-self.initial_state['alpha']) * self.state.l_response + self.initial_state['alpha'] * exog_info['new_patients']/(self.state.potential_pop + decision.enroll)
		new_succ = self.state.success + exog_info['succ_count']
		new_fail = self.state.failure + (exog_info['new_patients'] - exog_info['succ_count'])
		return {"potential_pop": enroll_pop,
				"success": new_succ,
				"failure": new_fail,
				"l_response": new_lambda}

	def objective_fn(self, decision):
		"""
        computes contribution of enrollments
        :param decision: namedtuple - contains all decision info
        :param exog_info: contains all exogenous info
        :return: float - calculated contribution
        """
		obj_part = (1-decision.prog_continue) * decision.drug_success * self.initial_state['success_rev'] - decision.prog_continue * (self.initial_state['program_cost'] + self.initial_state['patient_cost'] * decision.enroll)
		return obj_part
	
	def step(self, decision):
		"""
        steps the process forward by one time increment by updating the sum of the contributions, the
        exogenous information, and the state variable
        :param decision: namedtuple - contains all decision info
        :return: none
        """
		exog_info = self.exog_info_fn(decision)
		self.objective += self.objective_fn(decision)
		exog_info.update(self.transition_fn(decision, exog_info))
		self.state = self.build_state(exog_info)
		
if __name__ == "__main__":
	# this is an example of creating a model, using a random policy, and running until the drug is declared a success/failure or 
	# we reach the maximum number of trials
	t = 0
	stop = False
	# extracts data from given data set; defines initial state
	file = 'Trials Parameters.xlsx'
	raw_data = pd.ExcelFile(file)
	data = raw_data.parse('Exogenous Data')
	state_variables = ['potential_pop', 'success', 'failure', 'l_response']
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
	
	while t <= initial_state['trial_size'] and stop == False:
		p_belief = M.state.success / (M.state.success + M.state.failure)
		# drug_success = 1 if successful, 0 if failure, -1 if continue trial
		if p_belief > initial_state['theta_stop_high']:
			decision = {'prog_continue': 0, 'drug_success': 1}
			stop = True
		elif p_belief < initial_state['theta_stop_low']:
			decision = {'prog_continue': 0, 'drug_success': 0}
			stop = True
		else:
			decision = {'prog_continue': 1, 'drug_success': -1}
		decision['enroll'] = np.random.choice(range(initial_state['enroll_min'], initial_state['enroll_max']+initial_state['enroll_step'], initial_state['enroll_step'])) if stop == False else 0
		x = M.build_decision(decision)
		print("t={}, obj={}, state.potential_pop={}, state.success={}, state.failure={}, x={}".format(t, M.objective, M.state.potential_pop, M.state.success, M.state.failure, x))
		M.step(x)
		t += 1
		
	print("\nStopping state: ")			
	print("t={}, obj={}, state.potential_pop={}, state.success={}, state.failure={}, x={}".format(t, M.objective, M.state.potential_pop, M.state.success, M.state.failure, x))
	
	pass