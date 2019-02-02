import numpy as np
import pandas as pd
from copy import copy
from collections import (namedtuple, defaultdict)
import matplotlib.pyplot as plt

from StaticModelAdaptive import StaticModel
from PolicyAdaptive import Policy



if __name__ == "__main__":

    seed = 89720123


    #reading parameter file and initializing variables
    file = 'Parameters.xlsx'
    parDf = pd.read_excel(file, sheet_name = 'parameters')
    parDict=parDf.set_index('Index').T.to_dict('list')
    print("Starting adaptive stochastic shortest path with parameters")
    params = {key:v for key, value in parDict.items() for v in value}




    params.update({'seed':seed})
    print(params)



    state_names = ['CurrentNode', 'CurrentNodeLinksCost']
    decision_names = ['NextNode']
    

    # create the model, given the above policy
    M = StaticModel(state_names, decision_names,  params)
    policy_names = ['PureExploitation']
    P = Policy(M, policy_names)
    
    theta_list = M.init_args['theta_set'].split()
    obj_along_iterations = {theta:[] for theta in theta_list}
    vbar_along_iterations = {theta:[] for theta in theta_list}


    for theta in theta_list:
        
        model = copy(M)
        model.theta_step = float(theta)
        model.prng = np.random.RandomState(model.init_args['seed'])
        
        print("************Starting iterations for theta {}".format(model.theta_step))

        for ite in list(range(model.init_args['nIterations'])):

            
            model.obj = 0 
            model.state = model.build_state(model.init_state)

            print("\tTheta {}  - Iteration {} - Stepsize {:.2f} - InitState {} - Vbar {:.2f}".format(model.theta_step,model.n,model.alpha(),model.state.CurrentNode,model.V_t[model.state.CurrentNode]))

            step = 1
            while model.state.CurrentNode != model.init_args['target_node']:
                # calling policy and choosing a decision
                decision,vhat = P.make_decision(model)
                x = model.build_decision({'NextNode': decision})
                
                print("\t\t Theta {}  - Iteration {} - Step {} - Current State {} - vbar = {:.2f}".format(model.theta_step,model.n,step, model.state.CurrentNode,model.V_t[model.state.CurrentNode]))
                

                # update vfa
                vbar = model.update_VFA(vhat)
                
                # print current state, decision , vhat and new vbar
                #model.print_State()
                print("\t\tDecision={}, vhat {:.2f} and new vbar for current state {:.2f}".format(x[0],vhat,model.V_t[model.state.CurrentNode]))

                # transition to the next state w/ the given decision
                model.transition_fn(x)
                step += 1

            print("Finishing Theta {} and Iteration {} with {} steps. Total cost: {} \n".format(model.theta_step,model.n,step-1,model.obj))
            model.n+=1
            obj_along_iterations[theta].append(model.obj)
            vbar_along_iterations[theta].append(model.V_t[model.origin_node])


        
        
    
    

        
    #Ploting the results
    fig1, axsubs = plt.subplots(1,2)
    fig1.suptitle('Comparison of theta^step - stepsize type {} - origin {}, target {}, dist {} '.format(M.init_args['stepsize_rule'],M.origin_node,M.target_node,M.dist) )
  
    color_list = ['b','g','r','m','c']
    nThetas = list(range(len(theta_list)))
    Iterations = list(range(model.init_args['nIterations']))

    
    totals = [np.array(obj_along_iterations[theta]).sum() for theta in theta_list]
    print("Totals ",totals)


    for theta,t in zip(theta_list,nThetas):
        

        axsubs[0].plot(Iterations, np.array(obj_along_iterations[theta]).cumsum(), "{}-".format(color_list[t]),label = "{}".format(theta))
        #axsubs[0].plot(Iterations, obj_along_iterations[theta], "{}-".format(color_list[t]),label = "{}".format(theta))

        axsubs[0].set_title('Objective function')
        axsubs[0].legend()
        axsubs[0].set_xlabel('Iterations')
        axsubs[0].set_ylabel('Cost')

        axsubs[1].plot(Iterations, vbar_along_iterations[theta], "{}o-".format(color_list[t]),label = "{}".format(theta))
        axsubs[1].set_title('Vbar')
        axsubs[1].legend()
        axsubs[1].set_xlabel('Iterations')
        axsubs[1].set_ylabel('Cost')


        #axsubs.plot(Iterations, obj_along_iterations[theta], "{}o-".format(color_list[t]),label = "{}".format(theta))
        #axsubs[0].set_title('Cost')
        #axsubs.legend()
        #axsubs.set_xlabel('Iterations')
        #axsubs.set_ylabel('Cost')

    plt.show()

    
    
    