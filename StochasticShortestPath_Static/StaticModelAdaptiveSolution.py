"""
Stochastic Shortest Path Extension
Using point estimates

"""
from collections import (namedtuple, defaultdict)

import numpy as np


class StaticModel():
    """
    Base class for model
    """

    def __init__(self, state_names, x_names, params):
        """
        Initializes the model

        :param state_names: list(str) - state variable dimension names
        :param x_names: list(str) - decision variable dimension names
        :param s_0: dict - need to contain at least information to populate initial state using s_names
        :param exog_info_fn: function -
        :param transition_fn: function -
        :param objective_fn: function -
        :param seed: int - seed for random number generator
        """

        self.init_args = params
        self.prng = np.random.RandomState(self.init_args['seed'])
        self.state_names = state_names
        self.x_names = x_names
        self.State = namedtuple('State', state_names)
        self.Decision = namedtuple('Decision', x_names)

        # Creating the graph and computing V_0
        self.g , self.V_t, self.origin_node, self.target_node, self.dist = self.createStochasticGraph()
        self.init_args.update({'start_node':self.origin_node,'target_node':self.target_node})

        self.exog_info = self.g

        #Constructing the initial state
        self.init_state =  {'CurrentNode': self.init_args['start_node'], 'CurrentNodeLinksCost': self.exog_info_fn(self.init_args['start_node'])}
        self.state = self.build_state(self.init_state)
        print("Initial State")
        self.print_State()
        
        # value of objective function
        self.obj = 0.0
        
        # current iteration
        self.n = 1
        #The stepsize will be set outside the constructor
        self.theta_step = 1

        # policy function, given by Bellman's equation
        self.policy = None


        

    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        return self.Decision(*[info[k] for k in self.x_names])
    
    def print_State(self):
        print(" CurrentNode: {} and costs on its edges: ".format(self.state.CurrentNode))
        print(printFormatedDict(self.state.CurrentNodeLinksCost))
    
    def update_VFA(self,vhat):
        self.V_t[str(self.state.CurrentNode)] = (1-self.alpha())*self.V_t[str(self.state.CurrentNode)] + self.alpha()*vhat
        return self.V_t[str(self.state.CurrentNode)]

    def exog_info_fn(self, i):
        return {j:self.prng.uniform(self.g.lower[(i,j)], self.g.upper[(i,j)]) for j in self.g.edges[i]}
    

    def transition_fn(self, decision):
        self.obj = self.obj + self.state.CurrentNodeLinksCost[decision.NextNode]
        self.state = self.build_state({'CurrentNode': decision.NextNode, 'CurrentNodeLinksCost': self.exog_info_fn(decision.NextNode)})
        return self.state

    def objective_fn(self):
        return self.obj



    def createStochasticGraph(self):
        # create a random graph of n nodes and make sure there is a feasible path from node '0' to node 'n-1' 
        g = randomgraphChoice(self.prng,self.init_args['nNodes'], self.init_args['probEdge'],self.init_args['LO_UPPER_BOUND'],self.init_args['HI_UPPER_BOUND'])
        print("Created the graph")  

        maxSteps = 0
        max_origin_node = None
        max_target_node = None
        for target_node in g.nodes:
            # find the max number of steps bewteen to the target_node and the origin_node that achieves that
            max_node,max_dist = g.truebellman(target_node)
            
            if max_dist > maxSteps:   
                maxSteps = max_dist
                max_origin_node = max_node
                max_target_node = target_node

        print("max_origin_node: {} -  max_target_node: {}  - distance: {}".format(max_origin_node,max_target_node,maxSteps))

        V_0 = g.bellman(max_target_node)

        print("Computed V_0")
        print(printFormatedDict(V_0))

        return g,V_0,max_origin_node,max_target_node,maxSteps




    def alpha(self):
        if self.init_args['stepsize_rule']=='Constant':
            return self.theta_step
        else:
            return self.theta_step/(self.theta_step + self.n - 1)  

    




# Stochastic Graph class 
class StochasticGraph:
    def __init__(self):
        self.nodes = list()
        self.edges = defaultdict(list)
        self.lower = {}
        self.distances = {}
        self.upper = {}

    def add_node(self, value):
        self.nodes.append(value)
    
    # create edge with normal weight w/ given mean and var
    def add_edge(self, from_node, to_node, lower, upper):
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = 1
        self.lower[(from_node, to_node)] = lower
        self.upper[(from_node, to_node)] = upper
    
    # return the expected length of the shortest paths w.r.t. given node
    def bellman(self, target_node):
        inflist = [np.inf]*len(self.nodes)
        # vt - value list at time t for all the nodes w.r.t. to target_node
        vt = {k: v for k, v in zip(self.nodes, inflist)}
        vt[target_node] = 0
        
        # decision function for nodes w.r.t. to target_node
        dt = {k:v for k,v in zip(self.nodes, self.nodes)}
        
        # updating vt
        for t in range(1, len(self.nodes)):            
            for v in self.nodes:
                for w in self.edges[v]:
                    # Bellman' equation 
                    if (vt[v] > vt[w] + 0.5*(self.lower[(v,w)] + self.upper[(v,w)])):
                        vt[v] = vt[w] + 0.5*(self.lower[(v,w)] + self.upper[(v,w)])
                        dt[v] = w 
        # print(vt)
        # print(g.distances)
        return(vt)   
    
    def truebellman(self, target_node):
        inflist = [np.inf]*len(self.nodes)
        # vt - list for values at time t for all the nodes w.r.t. to target_node
        vt = {k: v for k, v in zip(self.nodes, inflist)}
        vt[target_node] = 0
        
        # decision function for nodes w.r.t. to target_node
        dt = {k:v for k,v in zip(self.nodes, self.nodes)}
        
         # updating vt
        for t in range(1, len(self.nodes)):            
            for v in self.nodes:
                for w in self.edges[v]:
                    # Bellman' equation 
                    if (vt[v] > vt[w] + self.distances[(v, w)]):
                        vt[v] = vt[w] + self.distances[(v, w)]
                        dt[v] = w 
        # print(vt)
        # print(g.distances)

        v_aux = {k:-1 if v == np.inf else v for k,v in vt.items()}
        max_node = max(v_aux, key=v_aux.get)
        max_dist = v_aux[max_node]

        return(max_node,max_dist)  
      

 
def randomgraphChance(prng, n, p,LO_UPPER_BOUND,HI_UPPER_BOUND):
    g = StochasticGraph()
    for i in range(n):
        g.add_node(str(i))
    for i in range(n):
        for j in range(n):
            q = prng.uniform(0,1)
            if (i != j and q < p):
                lo = prng.uniform(0, LO_UPPER_BOUND)
                hi = prng.uniform(lo, HI_UPPER_BOUND)
                g.add_edge(str(i), str(j), lo, hi)
    return(g)


def randomgraphChoice(prng, n, p,LO_UPPER_BOUND,HI_UPPER_BOUND):
    g = StochasticGraph()
    for i in range(n):
        g.add_node(str(i))
    for i in range(n-p-2):
        edge_set = list(prng.choice(np.arange(0,n-1), p, replace=False))
        for add_neighbor in list(range(1)):
            neighbor = min(i+add_neighbor+1,n-1)
            if  not neighbor in edge_set:
                edge_set = edge_set + [neighbor]
        for j in  edge_set:  
            if (i != j):
                lo = prng.uniform(0, LO_UPPER_BOUND)
                hi = prng.uniform(lo, HI_UPPER_BOUND)
                g.add_edge(str(i), str(j), lo, hi)
    return(g)


    

def printFormatedDict(dictInput):
    nodeList = [int(node) for node in dictInput.keys()] 
    nodeList = sorted(nodeList)

    for node in nodeList:
        print("\t\tkey_{} = {:.2f}".format(str(node),dictInput[str(node)]))



