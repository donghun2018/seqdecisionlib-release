
import numpy as np
import cvxopt
from collections import (namedtuple, defaultdict)


def initLPMatrices(params,Bld_Net):
    #Initializing the matrix for the LP
    A = np.zeros((params['NUM_BLD_NODES'], params['NUM_BLD_NODES']*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])))
    for i in range(params['NUM_BLD_NODES']):
        for j in range(params['NUM_BLD_NODES']*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])):
            if (j < (i+1)*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])) and (j >= i*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])):
                #Checking for feasibility
                k=j-i*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])
                if (k<params['NUM_DEM_NODES']):
                    bloodnode = Bld_Net.bloodnodes[i]
                    demandnode = Bld_Net.demandnodes[k]
                    if (demandnode[2] == False and bloodnode[0] == demandnode[0]) or (demandnode[2] == True and params['SubMatrix'][(bloodnode[0], demandnode[0])] == True):
                        A[i,j] = 1
                else:   
                    A[i,j] = 1    


    G = np.zeros((params['NUM_DEM_NODES'] + 2*params['NUM_BLD_NODES']*params['NUM_PARALLEL_LINKS'] + params['NUM_DEM_NODES']*params['NUM_BLD_NODES'], params['NUM_BLD_NODES']*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])))
    # ineq constr for sum x_tbd < D_td
    for i in range(params['NUM_DEM_NODES']):
        for j in range(params['NUM_BLD_NODES']*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])):
            if (j % (params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS']) == i):
                G[i,j] = 1.
    # ineq constr for parallel links <= SLOPE_CAPAC
    for i in range(params['NUM_BLD_NODES']):
        for j in range(params['NUM_PARALLEL_LINKS']):
            G[params['NUM_DEM_NODES'] + i*params['NUM_PARALLEL_LINKS'] + j, (params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])*i + params['NUM_DEM_NODES'] + j] = 1

    # ineq constr for x_tbd >= 0
    for i in range(params['NUM_BLD_NODES']):
        for j in range(params['NUM_DEM_NODES']):
            G[params['NUM_DEM_NODES'] + params['NUM_BLD_NODES']*params['NUM_PARALLEL_LINKS'] + i*params['NUM_DEM_NODES'] + j, (params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])*i + j] = -1


    # ineq constr for x_parallel >= 0
    for i in range(params['NUM_BLD_NODES']):
        for j in range(params['NUM_PARALLEL_LINKS']):
            G[params['NUM_DEM_NODES'] + params['NUM_BLD_NODES']*params['NUM_PARALLEL_LINKS'] + params['NUM_DEM_NODES']*params['NUM_BLD_NODES'] + i*params['NUM_PARALLEL_LINKS'] + j,(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])*i + params['NUM_DEM_NODES'] + j] = -1


    h = np.ones(params['NUM_DEM_NODES'] + params['NUM_BLD_NODES']*params['NUM_PARALLEL_LINKS'])
    h[params['NUM_DEM_NODES']::params['NUM_PARALLEL_LINKS']]= params['SLOPE_CAPAC_LAST']
    h = np.append(h, np.zeros(params['NUM_BLD_NODES']*params['NUM_DEM_NODES'] + params['NUM_BLD_NODES']*params['NUM_PARALLEL_LINKS']))
        
    A = cvxopt.matrix(A)
    G = cvxopt.matrix(G)

    coeff = [np.concatenate((np.array(Bld_Net.demcontrib[bld]),np.zeros(params['NUM_PARALLEL_LINKS']))) if int(bld[1])< params['MAX_AGE']-1 else np.concatenate((np.array(Bld_Net.demcontrib[bld]),np.add(np.zeros(params['NUM_PARALLEL_LINKS']),params['DISCARD_BLOOD_PENALTY'])))  for bld in Bld_Net.bloodnodes]
    coeff = [ai for a in coeff for ai in a ]    
    coeff = np.array(coeff)

    return (A,G,h,coeff)

class Policy():
    """
    Base class for Static Stochastic Shortest Path Model policy
    """

    def __init__(self,params,Bld_Net):
        """
        Initializes the policy
        """

        self.A,self.G,self.h,self.coeff = initLPMatrices(params,Bld_Net)
    

    def getLPSol(self,params,M,iteration,t,solDemList,solHoldList,IS_TRAINING):
       
        c_t = [np.concatenate((np.multiply(np.array(M.Bld_Net.demcontrib[bld]),-1),np.multiply(M.Bld_Net.parallelarr[(t, bld, M.Bld_Net.supersink)],-params['DISCOUNT_FACTOR']))) if int(bld[1])< params['MAX_AGE']-1 else np.concatenate((np.multiply(np.array(M.Bld_Net.demcontrib[bld]),-1),np.add(np.multiply(M.Bld_Net.parallelarr[(t, bld, M.Bld_Net.supersink)],-params['DISCOUNT_FACTOR']),-params['DISCARD_BLOOD_PENALTY'])))  for bld in M.Bld_Net.bloodnodes]
        c = [ai for a in c_t for ai in a ]
        b = np.array(M.Bld_Net.bloodamount)
        self.h[:params['NUM_DEM_NODES']] = M.Bld_Net.demandamount

        
        c = cvxopt.matrix(c)
        b = cvxopt.matrix(b,size=(params['NUM_BLD_NODES'],1),tc='d')
        h = cvxopt.matrix(self.h)
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.lp(c, self.G, h, self.A, b,solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}}) 
        #sol = cvxopt.solvers.lp(c, self.G, h, self.A, b) 

        x = sol['x']
           
        x = np.array(x)
        x = np.squeeze(x)
        
        val = np.dot(x, self.coeff)
        
        
        
        xDem = [x[i*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS']):i*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])+params['NUM_DEM_NODES']] for i in list(range(params['NUM_BLD_NODES']))]
        xDemFlat = [xij for xi in xDem for xij in xi]
        solDemRec=(iteration,t,xDem.copy())
        solDemList.append(solDemRec)
        
        hld=[np.sum(x[i*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])+params['NUM_DEM_NODES']:(i+1)*(params['NUM_DEM_NODES']+params['NUM_PARALLEL_LINKS'])]) for i in list(range(params['NUM_BLD_NODES']))]
        solHoldRecord = (iteration,t,hld.copy())
        solHoldList.append(solHoldRecord)
        hld = np.array(hld)
        
        invByBlood = [np.sum(M.bld_inv[i*params['MAX_AGE']:(i+1)*params['MAX_AGE']]) for i in list(range(len(params['Bloodtypes']))) ]
        demByBlood = [np.sum(M.Bld_Net.demandamount[i*(len(params['Surgerytypes'])*len(params['Substitution'])):(i+1)*(len(params['Surgerytypes'])*len(params['Substitution']))]) for i in list(range(len(params['Bloodtypes']))) ]
        
        xDemFlat = [xij for xi in xDem for xij in xi]
        xDemMat = np.array(xDemFlat).reshape(params['NUM_BLD_NODES'],params['NUM_DEM_NODES'])
        xDemMatColSum =  xDemMat.sum(axis=0)   
        covByBlood = [ np.sum(xDemMatColSum[i*(len(params['Surgerytypes'])*len(params['Substitution'])):(i+1)*(len(params['Surgerytypes'])*len(params['Substitution']))]) for i in list(range(len(params['Bloodtypes']))) ]
        covByBlood = np.array(covByBlood).astype(int)
        
        hldByBlood = [int(np.sum(hld[i*params['MAX_AGE']:(i+1)*params['MAX_AGE']])) for i in list(range(len(params['Bloodtypes']))) ]
        disByBlood = hld[params['MAX_AGE']-1::params['MAX_AGE']]
        disByBlood = np.array(disByBlood)
        disByBlood = disByBlood.astype(int)
        
        if False:
            print('Iteration = ', iteration)
            print('Time period = ', t)
            print('Demand = ', np.sum(M.Bld_Net.demandamount))
            print('Supply = ', np.sum(M.Bld_Net.bloodamount))
            print('Blood Used = ', np.sum(M.bld_inv) - np.sum(hld))
            print('Blood Held = ', np.sum(hld))
            print('Inventory by BloodType ',invByBlood)
            print('Demand By BloodType ',demByBlood)
            print('Used By BloodType ', list(covByBlood))
            print('Hold By BloodType ',hldByBlood)
            print('Discard By BloodType ', list(disByBlood))
            print('Contribution = ', val)   
            print('Donation = ', np.sum(M.donation)) 
            print('\n')
        
        hld = hld.astype(int)
        
        if IS_TRAINING and params['IS_PERTUB']:
            epsilon = PERTUB_GEN.poisson(LAMBDA_PERTUB, params['NUM_BLD_NODES']) 
            signE = PERTUB_GEN.choice([-1,1], size=params['NUM_BLD_NODES'], replace=True, p=None)
            hld = hld+epsilon*signE
            hld = np.maximum(np.zeros(params['NUM_BLD_NODES']),hld)
            hld = hld.astype(int)
    
        # dual variables
        d = sol['y']
               
        return sol,val,x,hld,d,solDemList,solHoldList
    
    def updateVFAs(self,params,M,iteration,t,d, slopesList,updateVfaList):
        alpha = 0

        # set the dual variables to respective parallel arcs
        for i in range(params['NUM_BLD_NODES']):
            # put the value of the dual varible d[i+1] in the parallel arc, associated 
            # with the amount of resource in the inventory associated with holdnode[i]
            # the holdnodes with the oldest age do not get updated 
            
            recordSlopes = (iteration,t,M.Bld_Net.parallelarr[(t, M.Bld_Net.holdnodes[i], M.Bld_Net.supersink)].copy())
            slopesList.append(recordSlopes)
        
            index = M.bld_inv[i]
            if index>=0:
                if (t>0 and M.Bld_Net.holdnodes[i][1]<str(params['MAX_AGE']-1)):
                    vhat=d[i+1]
                    
                    if index >= params['NUM_PARALLEL_LINKS'] - 1:
                        index = params['NUM_PARALLEL_LINKS'] - 1
                        
                    arr = M.Bld_Net.varr[(t-1,M.Bld_Net.holdnodes[i], M.Bld_Net.supersink)]
                    sqGradArr = M.Bld_Net.sqGrad[(t-1,M.Bld_Net.holdnodes[i])]
                    

                    if iteration < params['NUM_ITER_STEP_ONE']:
                            alpha = 1
                    else:
                        if (params['STEPSIZE_RULE'] == 'C'):
                            alpha = params['ALPHA']
                        elif (params['STEPSIZE_RULE'] == 'A'):
                            sqGradArr[index] += np.power(vhat-arr[index],2) 
                            alpha = params['ETA']/(np.sqrt(sqGradArr[index]+params['STEP_EPS']))
                   
                    vbar = arr[index]
                    vnew = alpha*vhat +(1-alpha)*vbar
                    arr[index] = vnew
                    
                    recordUpdateVfa = (iteration,t-1,M.Bld_Net.holdnodes[i][0],M.Bld_Net.holdnodes[i][1],index,vhat,vbar,sqGradArr[index],alpha,vnew)
                    updateVfaList.append(recordUpdateVfa)

                    #Projecting back in case the vfa is not concave anymore
                    if (vnew>vbar): #Look to the left
                        indSetL=[i for i in list(range(0,index+1)) if arr[i]<=vnew]
                        if (len(indSetL)>0):
                            if params['PROJECTION_ALGO'] == 'Avg':
                                avg = np.mean(arr[indSetL])
                                arr[indSetL]=avg
                            elif params['PROJECTION_ALGO'] == 'Copy':
                                arr[indSetL]=vnew 
                            else:
                                if index > 0:
                                    j=index-1
                                    while (j>=0 and arr[j] < arr[j+1]):
                                        arr[j]= alpha*vhat +(1-alpha)*arr[j]
                                        j-=1
                                else:
                                    arr[index]=vnew 


                            
                    elif (vnew<vbar): #Look to the right
                        indSetR=[i for i in list(range(index,params['NUM_PARALLEL_LINKS'])) if arr[i]>=vnew]
                        if (len(indSetR)>0):
                            if params['PROJECTION_ALGO'] == 'Avg':
                                avg = np.mean(arr[indSetR])
                                arr[indSetR]=avg
                            elif params['PROJECTION_ALGO'] == 'Copy':   
                                arr[indSetR]=vnew
                            else:
                                if index < params['NUM_PARALLEL_LINKS']-1:
                                    j=index+1
                                    while (j<params['NUM_PARALLEL_LINKS'] and arr[j] > arr[j-1]):
                                        arr[j] = alpha*vhat +(1-alpha)*arr[j]
                                        j+=1
                                else:
                                    arr[index]=vnew 

        return alpha,slopesList,updateVfaList

