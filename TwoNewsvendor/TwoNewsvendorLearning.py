"""
Two Newsvendor as a Learning Problem 

Author: Andrei Graur

This program implements a model for the two newsvendor 
problem where the field agent and the central command
both view the problem of choosing the right bias to add
or substract as a learning problem. Run the code with the
python command, no arguments given. 

"""

import numpy as np
import pandas as pd
import math
import xlrd

from TwoNewsvendor import Model_Field
from TwoNewsvendor import Model_Central

# the class implementing the objects that represent the 
# avaiable choices of the two agents, which are the biases to add
class Choice:
	def __init__(self, quantity, util_estimate, W_precision_estimate, theta, nu_bar = 0.5):
		'''
		The function that initializes the choice object

		param: quantity - int: the quantity in units equal to the bias
		param: util_estimate - float: the estimate of what the utility will 
		be when we use this bias; we initialize it to 0 in main 
		param: precision_estimate - float: the estimate of what the 
		precision of next experiment of using this bias will be
		param: theta - float: the tunable parameter. It can be for the  UCB policy or for the IE polidy
		'''
		
		self.n = 0

		self.quantity = quantity
		self.util_estimate = util_estimate
		self.accumulated_precision = W_precision_estimate

		self.theta = theta
		

		#Variables to compute the variance of W
		self.W_precision = W_precision_estimate 
		self.W_variance = 1 / float(self.W_precision) 
		
		self.nu_bar = nu_bar
		self.W_bar = util_estimate
		self.W_beta = 0
		self.W_delta = 0
		self.W_lambda = 0
		self.nu = 1
		
		
		



	# the function that uploads the results of the experiment of trying
	# this bias and updates the corresponding beliefs about this choice
	def upload_results(self, W):
		self.n += 1

		self.nu = (self.nu)/(1+self.nu-self.nu_bar)
		self.W_beta = (1-self.nu)*self.W_beta + self.nu * (W - self.W_bar)
		self.W_delta = (1-self.nu)*self.W_delta + self.nu * ((W - self.W_bar)**2)
		
	
		# update the variance
		if self.n > 1:
			#self.W_variance = (((self.n - 2.0) / float(self.n - 1)) * self.W_variance +(1.0 / self.n) * ((W - self.util_estimate) ** 2))
			  
			self.W_variance = (self.W_delta - (self.W_beta**2))/(1+self.W_lambda)
			if self.W_variance < 0.0001:
				self.W_precision = 10
			else:
				self.W_precision = 1 / float(self.W_variance)
			
		alpha = self.W_precision / (self.accumulated_precision + self.W_precision)

		if self.n >1:
			self.W_lambda = ((1-alpha)**2)*self.W_lambda + (alpha)**2
		else:
			self.W_lambda = (alpha)**2
			
		self.W_bar = (1-alpha)*self.W_bar + alpha*W
		

		# update estimate and experiment precision 
		self.util_estimate = ((self.util_estimate * self.accumulated_precision +
						W * self.W_precision) / 
						(self.accumulated_precision + self.W_precision))
		self.accumulated_precision += self.W_precision


	# the function that returns the bias attribute of this object
	def get_choice_quantity(self):
		return self.quantity

	# the cost function approximation for this choice of bias
	def get_UCB_value(self, time):
		if self.n == 0:
			UCB_val =  np.inf

		else:
			UCB_val = (self.util_estimate + self.theta * math.sqrt(math.log(time) / self.n))
		return UCB_val

	def get_IE_value(self):
		
		IE_val = (self.util_estimate + self.theta * math.sqrt(1/self.accumulated_precision))
		return IE_val
	
	def get_nb_experiments(self):
		return self.n

	def getAllParametersHeaderList(self):
		outL="bias_choice n mu_bar_estimate Beta sigma IE_value UCB_value W_nu  W_beta W_delta  W_variance alpha W_bar W_lambda W_precision "
		return outL.split()

	def getAllParametersList(self,time):
		outL="{:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} ".format(self.quantity,self.n,self.util_estimate,self.accumulated_precision,math.sqrt(1/self.accumulated_precision),self.get_IE_value(),self.get_UCB_value(time),self.nu,self.W_beta,self.W_delta,self.W_variance,self.getOSA(),self.W_bar,self.W_lambda,self.W_precision)
		return outL.split()

	def getMainParametersHeaderList(self):
		outL="bias_choice n mu_bar_estimate Beta sigma W_bar W_precision W_variance "
		return outL.split()

	def getMainParametersList(self):
		return [self.quantity,self.n,self.util_estimate,self.accumulated_precision,math.sqrt(1/self.accumulated_precision),self.W_bar,self.W_precision,self.W_variance]

	def printChoiceParameters(self,n):
		valuesList = self.getAllParametersList(n)
		headerList = self.getAllParametersHeaderList()
		outStr=""
		for i in range(len(valuesList)):
			outStr += "{}: {}, ".format(headerList[i],valuesList[i])
		return outStr

# the model for the field agent treating the problem as a 
# learning problem 
class Learning_model_field(Model_Field):
	def __init__(self, theta, *args, **kwargs):
		super(Learning_model_field, self).__init__(*args, **kwargs)
		range_list = self.init_args['bias_interval_field'].split(",")
		range_list = [int(e) for e in range_list]
		self.choice_range =range(range_list[0],range_list[1]+1)
		self.resetModel(theta)

	def resetModel(self,theta):
		self.choices = {}	
		for value in self.choice_range:

			self.choices[value] = Choice(value, 0, 0.01, theta)

		super(Learning_model_field, self).resetModel(None)



	# the new transition function for the learning approach
	def transition_fn(self, exog_info):
		
		# update the results of having tried out the used choice 
		choice_used = self.choices[self.decision.bias_applied]
		
		#print("Field Choice state pre update")
		#outStr = choice_used.printChoiceParameters(self.n+1)
		#print(outStr)

		choice_used.upload_results(-self.pen_incurred)
		# update beliefs about the external source
		super(Learning_model_field, self).transition_fn(exog_info)
		
		#print("Field Choice state post update")
		#outStr = choice_used.printChoiceParameters(self.n+1)
		#print(outStr)

	def getMainParametersList(self):
		listPar = [self.choices[x].getMainParametersList() for x in self.choice_range]
		listParFlat = [elem for l in listPar for elem in l]
		return listParFlat

	def getMainParametersHeaderList(self):
		listPar = [self.choices[x].getMainParametersHeaderList() for x in self.choice_range]
		listParFlat = [str(x)+"_field_"+elem for x,l in zip(self.choice_range,listPar) for elem in l]
		return listParFlat

	def getMainParametersDf(self):
		dictPar = {x:self.choices[x].getMainParametersList() for x in self.choice_range}
		pdPar = pd.DataFrame(dictPar)
		pdPar = pdPar.transpose()
		pdPar.columns = self.choices[self.choice_range[0]].getMainParametersHeaderList()

		print(pdPar)
		return pdPar



# the model for the central command treating the problem as a 
# learning problem
class Learning_model_central(Model_Central):
	def __init__(self, theta, *args, **kwargs):
		super(Learning_model_central, self).__init__(*args, **kwargs)
		range_list = self.init_args['bias_interval_central'].split(",")
		range_list = [int(e) for e in range_list]
		self.choice_range=range(range_list[0],range_list[1]+1)
		self.resetModel(theta)

	def resetModel(self,theta):
		self.choices = {}
		for value in self.choice_range:

			self.choices[value] = Choice(value, 0, 0.01, theta)

		super(Learning_model_central, self).resetModel(None)


	def transition_fn(self, exog_info):
		# update the results of having tried out the used choice 
		choice_used = self.choices[self.decision.bias_applied]

		#print("Central Choice state pre update")
		#outStr = choice_used.printChoiceParameters(self.n+1)
		#print(outStr)
		
		choice_used.upload_results(-self.pen_incurred)
		# update beliefs about the external source
		super(Learning_model_central, self).transition_fn(exog_info)
		
		#print("Central Choice state pos update - W = {:.2f}".format(-self.pen_incurred))
		#outStr = choice_used.printChoiceParameters(self.n+1)
		#print(outStr)

	def getMainParametersList(self):
		listPar = [self.choices[x].getMainParametersList() for x in self.choice_range]
		listParFlat = [elem for l in listPar for elem in l]
		return listParFlat

	def getMainParametersHeaderList(self):
		listPar = [self.choices[x].getMainParametersHeaderList() for x in self.choice_range]
		listParFlat = [str(x)+"_central_"+elem for x,l in zip(self.choice_range,listPar) for elem in l]
		return listParFlat

	def getMainParametersDf(self):
		dictPar = {x:self.choices[x].getMainParametersList() for x in self.choice_range}
		pdPar = pd.DataFrame(dictPar)
		pdPar = pdPar.transpose()
		pdPar.columns = self.choices[self.choice_range[0]].getMainParametersHeaderList()

		print(pdPar)
		return pdPar




