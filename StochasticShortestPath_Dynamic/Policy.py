''' 

The code for the lookahead policy we use in our 
Static Model

'''

import numpy as np

# the lookahead policy
class LookaheadPolicy():
	def __init__(self, model):
		self.model = model
	

	# function returning the decision x_t from the current state 
	# and current time. The argument decisions is given to 
	# use a local variable rather than for getting outside information
	def get_decision(self,  METRIC):

		# the matrix with decisions to be made for each node and each time
		decisions = [ ([0] * self.model.G.vertexCount) for row in range(self.model.G.Horizon + 1) ]


		# initialize the value costs at different nodes at different times to infinity
		V = np.ones((self.model.G.Horizon + 1, self.model.G.vertexCount)) * np.inf
		# make the costs at the destination 0
		for t_prime in range(self.model.G.Horizon + 1):
			V[t_prime][self.model.G.end_node] = 0

		
		# the algortihm that uses the "stepping backwards in time" method
		lookAheadTime = self.model.G.Horizon - 1
		while lookAheadTime >= 0:
			for k in range(self.model.G.vertexCount):
				# find the solutions to Bellman's eq. that are shown 
				# in 5.22 and 5.23
				argMin = - 1
				minVal = np.inf
				for l in self.model.G.neighbors[k]:
					if (METRIC == "PERCENTILE"):
						spread = self.model.G.spreads[k][l]
						mean = self.model.estimated_costs[k][l]
						if minVal >= V[lookAheadTime + 1][l] + self.use_percentile_val(self.model.theta, spread, mean):		
							argMin = l
							minVal = V[lookAheadTime + 1][l] + self.use_percentile_val(self.model.theta, spread, mean)
					else:
						if minVal >= V[lookAheadTime + 1][l] + dist[k][l]:		
							argMin = l
							minVal = V[lookAheadTime + 1][l] + dist[k][l]

				# updating the solutions to the equations
				V[lookAheadTime][k] = minVal
				decisions[lookAheadTime][k] = argMin
			lookAheadTime -= 1
		

		return decisions[0][self.model.state.node]


	'''
	the function that takes as arguments the percentile we are going to
	use, theta (espressed as a value in [0,1]), the spread for a link and
	the mean cost of that link and returns the value corresponding to
	the theta precentile of the interval [(1 - spread) * mean, (1 + spread) * mean]
	'''
	def use_percentile_val(self,theta, spread, mean):
		point_val = 1 - spread + (2 * spread) * theta
		used_cost = mean * point_val
		return used_cost

