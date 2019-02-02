''' 
This program generates a graph with 35 vertices that will be used for 
comparing the dynamic and static lookahead approaches on it. 

Run without any arguements. 

Author: Andrei Graur 

'''

import numpy as np
import networkx as nx
import pandas as pd
import math
from collections import (namedtuple, defaultdict)


class GraphGenerator():
	"""
	Base class for the static model
	"""

	def __init__(self, params):
		self.init_args = params
		self.prng = np.random.RandomState(params['seed'])
		self.meanCosts = defaultdict(dict)
		self.dist= defaultdict(dict)
		self.spreads = defaultdict(dict)
		self.neighbors = defaultdict(list)
		self.vertices = []
		
		
		# The start and end node will change based on the network graph that is going to be constructed - we are going to select the pair with the longest shortest path
		self.start_node = 0
		self.end_node = 0 
		self.steps = 0
		self.vertexCount = 1
		self.Horizon = self.vertexCount + 1
		self.mPathsList=[]
		self.nPaths = 0
	

	def createNetworkSteps(self):

		filename = 'Network_Steps.xlsx'
		nSteps = self.init_args['nSteps']

		G = nx.DiGraph()

		nodeCount = 0
		nodesPerLevel = defaultdict(list)
		midGraph = math.ceil(nSteps/2)
		
		for level in range(nSteps):
			if level < midGraph:
				nNodes = level * 2 + 1
			else:
				nNodes = (nSteps-level-1) * 2 + 1

			for i in range(nNodes):	
				nodesPerLevel[level].append(nodeCount) 
				nodeCount += 1

		for level in range(nSteps-1):
			for i in nodesPerLevel[level]:
				G.add_node(i)
				edge_set = list(self.prng.choice(nodesPerLevel[level+1], min(3,len(nodesPerLevel[level+1])), replace=False))
				for j in edge_set:
					meanWeight = 1
					G.add_edge(i, j, weight = meanWeight)

		self.construct_network_objects(G,filename,0,nodeCount - 1)

		
		return 1




	def get_deadline(self):
		return (self.init_args['costMin']+ (self.init_args['costMax'] - self.init_args['costMin'])*self.init_args['deadlinePerc']) * (self.steps)


	def get_avg_cost_paths(self,shouldPrintPaths=False):
		#Printing the length and the costs of all paths
		totalCostList = []
		if shouldPrintPaths:
			print("*************Printing the length and the costs of all paths************")
		p=0
		for path in self.mPathsList:
			nSteps = len(path)
			totalCost = 0
			p += 1
			pathString = 'Path {}:  '.format(p)
			for n in range(nSteps-1):
				fromNode = path[n]
				toNode = path[n+1]
				totalCost += self.meanCosts[fromNode][toNode]
				#edge = " ({}, {}, {:.2f}, {:.2f}) ".format(fromNode,toNode,self.meanCosts[fromNode][toNode],totalCost) 
				#pathString += edge
			pathString += " - {} steps and {:.2f} total mean cost".format(nSteps,totalCost)
			totalCostList.append(totalCost)
			if shouldPrintPaths:
				print(pathString)
			avgTotalCost = np.array(totalCostList).mean()
		return avgTotalCost


	def construct_network_objects(self,G,filename,start_node,end_node):
		size = G.number_of_nodes()
		recordList = []
		for fromNode in range(size):
			self.vertices.append(fromNode)

			
			for toNode in G.neighbors(fromNode):

				self.neighbors[fromNode].append(toNode)
				self.meanCosts[fromNode][toNode] = self.prng.uniform(self.init_args['costMin'], self.init_args['costMax'])
				self.spreads[fromNode][toNode] = self.prng.uniform(0, self.init_args['maxSpreadPerc'])
				self.dist[fromNode][toNode] = 1

				record = (fromNode,toNode,self.meanCosts[fromNode][toNode], self.spreads[fromNode][toNode],size)
				recordList.append(record)

		if (self.init_args['printGraph']):
			headerDf = ['From','To','Cost','Spread','Graph_size']
			df = pd.DataFrame.from_records(recordList,columns=headerDf)
			df.to_excel(filename, sheet_name = 'Network', index = False)

		self.start_node = start_node
		self.end_node = end_node
		self.steps = nx.shortest_path_length(G, start_node, end_node)
		self.vertexCount = size
		self.Horizon = self.vertexCount + 1
		
		
		# We need to add the dummy link of cost 0 to the destination node	
		r = self.end_node
		self.spreads[r][r] = 0
		self.neighbors[r].append(r)
		self.meanCosts[r][r] = 0
		self.dist[r][r] = 0


		self.mPathsList = list(nx.all_simple_paths(G, source=self.start_node, target=self.end_node))
		self.nPaths=len(self.mPathsList)



	def createNetworkChance(self):

		filename = 'Network_Chance.xlsx'
		chance = self.init_args['edgeProb']
		size = self.init_args['nNodes']
		

		G = nx.DiGraph()
		nbIterations = 0
		done = 0

		while done == 0:
			for i in range(size):
				G.add_node(i)


			for i in range(size):
				for j in range(size):
					if self.prng.uniform() < chance:
						if i != j:
						    meanWeight = 1
						    G.add_edge(i, j, weight = meanWeight)
									
			maxLength = 0
			mSource = None
			mDest = None
			mPaths = 0
			mPathsList = []

			breakLoop = False
			
			for i in range(size): 
				for j in range(size):
					if nx.has_path(G, i, j):
						length = nx.shortest_path_length(G, i, j)
						if (length >= maxLength):	
							paths = list(nx.all_simple_paths(G, source=i, target=j))
							nPaths=len(paths)

							if (nPaths > mPaths):				
								maxLength = length
								mSource = i
								mDest = j
								mPaths = nPaths
								mPathsList = paths

							if (length > self.init_args['lengthThreshold']):
								breakLoop=True
								break
					else:
						pass;
				if breakLoop:
					break

			print("Iteration {}, Source {}, Dest {}, Length {}, number of paths {}".format(nbIterations,mSource,mDest,maxLength,mPaths))

			if maxLength > self.init_args['lengthThreshold'] and mPaths > self.init_args['numberPathsThreshold']:
				# the graph is good and we will use it and stop the loop
				done = 1
				self.construct_network_objects(G,filename,mSource,mDest)
			else:
				G.clear()
				nbIterations += 1


		return nbIterations+1
