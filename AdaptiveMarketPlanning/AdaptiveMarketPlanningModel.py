"""
Adaptive Market Planning Model class

Adapted from code by Donghun Lee (c) 2018

"""
		
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

class AdaptiveMarketPlanningModel():
	"""
	Base class for model
	"""

	def __init__(self, state_names, x_names, s_0, T,reward_type,price = 1.0, cost = 1.0, exog_info_fn=None, transition_fn=None, objective_fn=None, seed=20180613):
		"""
		Initializes the model

		:param state_names: list(str) - state variable dimension names
		:param x_names: list(str) - decision variable dimension names
        :param s_0: dict - need to contain at least information to populate initial state using s_names
		:param price: float - price p
		:param cost: float - cost c
		:param exog_info_fn: function - calculates relevant exogenous information
		:param transition_fn: function - takes in decision variables and exogenous information to describe how the state
			   evolves
		:param objective_fn: function - calculates contribution at time t
        :param seed: int - seed for random number generator
        """
		
		self.init_args = {seed: seed}
		self.prng = np.random.RandomState(seed)
		self.init_state = s_0
		self.T = T
		self.reward_type = reward_type
		self.state_names = state_names
		self.x_names = x_names
		self.State = namedtuple('State', state_names)
		self.state = self.build_state(s_0)
		self.Decision = namedtuple('Decision', x_names)
		self.obj = 0.0
		self.past_derivative = 0.0
		self.cost = cost
		self.price = price
		self.t = 0
		self.learning_list=[]



	# this function gives a state containing all the state information needed
	def build_state(self, info):
		return self.State(*[info[k] for k in self.state_names])

	# this function gives a decision 
	def build_decision(self, info):
		return self.Decision(*[info[k] for k in self.x_names])

	# this function gives the exogenous information that is dependent on a random process
	# computes the f_hat, chnage in the forecast over the horizon
	def exog_info_fn(self, decision):
		# return new demand based on a given distribution
		return {"demand": self.prng.exponential(100)}

	# this function takes in the decision and exogenous information to return
	# new state
	def transition_fn(self, decision, exog_info):
		
		self.learning_list.append(self.state.order_quantity)

		# compute derivative
		derivative = self.price - self.cost if self.state.order_quantity < exog_info['demand'] else - self.cost
		# update order quantity
		new_order_quantity = max(0, self.state.order_quantity + decision.step_size * derivative)
		print(' step ', decision.step_size)
		print(' derivative ', derivative)
		# count number of times derivative changes sign
		new_counter = self.state.counter + 1 if self.past_derivative * derivative < 0 else self.state.counter
		self.past_derivative = derivative



		return {"order_quantity": new_order_quantity, "counter": new_counter}

	# this function calculates how much money we make
	def objective_fn(self, decision, exog_info):
		self.order_quantity=self.state.order_quantity
		obj_part = self.price * min(self.order_quantity, exog_info['demand']) - self.cost * self.state.order_quantity
		return obj_part

	# this method steps the process forward by one time increment by updating the sum of the contributions, the
	# exogenous information and the state variable
	def step(self, decision):
		self.t_update()
		exog_info = self.exog_info_fn(decision)
		onestep_contribution = self.objective_fn(decision, exog_info)

		print("t {}, Price {}, Demand {}, order_quantity {}, contribution {}".format(self.t,self.price,exog_info['demand'],self.order_quantity,onestep_contribution))
		
		#Check if cumulative or terminal reward
		if (self.reward_type == 'Cumulative'):
			self.obj += onestep_contribution
		else:
			if (self.t == self.T):
				self.obj = onestep_contribution

		
		transition_info = self.transition_fn(decision, exog_info)
		self.state = self.build_state(transition_info)
		
		
		
	# Update method for time counter
	def t_update(self):
		self.t += 1
		return self.t