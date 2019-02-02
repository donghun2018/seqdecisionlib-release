

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time
import cvxopt
from collections import (namedtuple, defaultdict)
import os.path
import os
#from mpl_toolkits.mplot3d import Axes3D
#from memory_profiler import memory_usage


#from BloodManagementParsAndInitialState import *
from BloodManagementNetwork import *
from BloodManagementModel import *
from BloodManagementPolicy import initLPMatrices,Policy

def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.get_memory_info().rss


def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before,
            elapsed_time))
        return result
    return wrapper


def printParams(params):
    print(params) 
        

def loadParams(filename):

    parDf = pd.read_excel(filename, sheet_name = 'Parameters')
    parDict=parDf.set_index('Index').T.to_dict('list')
    params = {key:v for key, value in parDict.items() for v in value}

    params['PRINT']=False
    params['PRINT_ALL']=False
    params['OUTPUT_FILENAME'] = 'DetailedOutput.xlsx'

    params['SHOW_PLOTS']=False




    #Set here bloodtypes and substitutions that are allowed
    params['Bloodtypes'] = ['AB+', 'AB-', 'A+', 'A-','B+', 'B-', 'O+', 'O-'] 
    params['NUM_BLD_TYPES'] = len(params['Bloodtypes'])

    b = [(x,y) for x in params['Bloodtypes'] for y in params['Bloodtypes']]
    f = [False]*(len(params['Bloodtypes'])*len(params['Bloodtypes']))
    c = {k:v for k,v in zip(b, f)}
    #In case we want to allow subs
    c[('AB+', 'AB+')] = True

    c[('AB-', 'AB+')] = True
    c[('AB-', 'AB-')] = True

    c[('A+', 'AB+')] = True
    c[('A+', 'A+')] = True

    c[('A-', 'AB+')] = True
    c[('A-', 'AB-')] = True
    c[('A-', 'A+')] = True
    c[('A-', 'A-')] = True

    c[('B+', 'AB+')] = True
    c[('B+', 'B+')] = True

    c[('B-', 'AB+')] = True
    c[('B-', 'AB-')] = True
    c[('B-', 'B+')] = True
    c[('B-', 'B-')] = True

    c[('O+', 'AB+')] = True
    c[('O+', 'A+')] = True
    c[('O+', 'B+')] = True
    c[('O+', 'O+')] = True

    c[('O-', 'AB+')] = True
    c[('O-', 'A+')] = True
    c[('O-', 'B+')] = True
    c[('O-', 'O+')] = True
    c[('O-', 'AB-')] = True
    c[('O-', 'A-')] = True
    c[('O-', 'B-')] = True
    c[('O-', 'O-')] = True
    params['SubMatrix'] = c

    # Set here max age of blood
    params['MAX_AGE'] = 3
    params['Ages'] = list(range(params['MAX_AGE']))

    params['NUM_BLD_NODES'] = params['NUM_BLD_TYPES'] * params['MAX_AGE']

    # Set here blood demand nodes
    params['Surgerytypes'] = ['Urgent', 'Elective']
    params['Substitution'] = [True]

    params['NUM_SUR_TYPES'] = len(params['Surgerytypes'])
    params['NUM_DEM_NODES'] = params['NUM_BLD_TYPES'] * params['NUM_SUR_TYPES'] * len(params['Substitution'])


    # Solver params
    params['SLOPE_CAPAC_LAST'] = 100000
    params['MIN_CONST'] = 0.01
    params['EPSILON'] = 0.001


    # Set here number of iterations and time periods
    params['NUM_TRAINNING_ITER'] = int(params['NUM_TRAINNING_ITER'])
    params['NUM_TESTING_ITER']=int(params['NUM_TESTING_ITER'])
    params['NUM_ITER'] = int(params['NUM_TESTING_ITER'] + params['NUM_TRAINNING_ITER']) #Total number of iterations
    params['MAX_TIME']=int(15)
    params['Times'] = list(range(params['MAX_TIME']))

    # Set here VFA parameters
    # - If USE_VFA is set to True we are going to use VFA's when making the decisions - 
    # - If USE_VFA is set to False, it means that a  MYOPIC policy is going to be considered and all the parameters
    #related to VFA's (such as DISCOUNT_FACTOR, LOAD_VFA, SAVE_VFA, STEPSIZE_RULE, PROJECTION_ALGO, 
    #IS_PERTUB,SEED_TRAINING are ignored)
    #params['USE_VFA'] = True #If set to True we are going to use VFA's when making the decisions - False means a MYOPIC policy
    params['DISCOUNT_FACTOR'] = 0.95

    params['LOAD_VFA'] = False #If set to True we are going to initialize the VFA's with VFA's from previous runs - instead of all zeros
    params['NAME_LOAD_VFA_PICKLE'] = "Bld_Net10_P_C_Subs.pickle"
    params['SAVE_VFA'] = False #If we want to save/update the VFA's to be used in future runs
    params['NAME_SAVE_VFA_PICKLE'] = "Bld_Net10_P_C_Subs.pickle"

    # Set here the stepsize parameters
    params['STEPSIZE_RULE'] = 'C' #Possible values: 'C' for Constant or 'A' for AdaGrad
    params['NUM_ITER_STEP_ONE'] = 0 #Number of iterations with stepsize one 

    # Set here the CONSTANT stepsize parameter (not considered if AdaGrad stepsize is being used)
    #params['ALPHA'] = 0.2 #the stepsize for the other iterations

    #Set here the AdaGrad stepsize parameters (not considered if Constant stepsize is being used)
    params['STEP_EPS'] = 0.00000001
    params['ETA'] = 1

    # Set here the algorithm that should be use for projection back the slopes that break concavity
    # Possible algorithms for projecting back the slopes to enforce concavity are:
    # - 'Avg' to average the slopes that break concavity; \
    # - 'Copy' to copy the newly updated vbar to the slopes that break concavity
    # - 'Up' to update the slopes that break concavity with the current stepsize and vhat 
    params['PROJECTION_ALGO'] = 'Up' 


    #Perturb the solution during training iterations for exploration
    params['IS_PERTUB'] = False
    params['LAMBDA_PERTUB'] = 1
    params['PERTUB_GEN'] = np.random.RandomState(13247)

    # Set here one step contribution function parameters  - BONUSES and PENALTIES 
    params['AGE_BONUS']=np.zeros(params['MAX_AGE'])
    # params['AGE_BONUS']=[2]*MAX_AGE
    # params['AGE_BONUS']=list(reversed(list(range(0,MAX_AGE))))
    # params['AGE_BONUS']=list(range(0,MAX_AGE))
    # params['AGE_BONUS']=[0.5, 2] #It has to be the same length as MAX_AGE
    
    
    params['INFEASIABLE_SUBSTITUTION_PENALTY'] = -50
    params['NO_SUBSTITUTION_BONUS'] = 5
    params['URGENT_DEMAND_BONUS'] = 30
    params['ELECTIVE_DEMAND_BONUS'] = 5
    params['DISCARD_BLOOD_PENALTY'] = -10 #applied for the oldest age in the holding/vfa arcs


    # Set here Random Seeds 
    params['SEED_TRAINING'] = 1090377
    params['SEED_TESTING'] = 8090373

    #Set here the distribution for demand/donation/initial inventory
    params['SAMPLING_DIST'] = 'P' #Possible values: 'P' for Poisson or 'U' for uniform
    params['POISSON_FACTOR'] = 1

    # Set here max demand by blood type (when 'U'niform dist) or mean demand (when 'P'oisson dist) 
    params['DEFAULT_VALUE_DIST'] = 20
    d = [params['DEFAULT_VALUE_DIST']] * params['NUM_BLD_TYPES']
    params['MAX_DEM_BY_BLOOD'] = {k:v for k,v in zip(params['Bloodtypes'], d)}
    params['MAX_DON_BY_BLOOD'] = {k:v for k,v in zip(params['Bloodtypes'], d)}


    # Set here demand by blood type (for blood types that are different than the params['DEFAULT_VALUE_DIST'])
    params['MAX_DEM_BY_BLOOD']['AB+'] = 3
    params['MAX_DEM_BY_BLOOD']['B+'] = 9
    params['MAX_DEM_BY_BLOOD']['O+'] = 18
    params['MAX_DEM_BY_BLOOD']['B-'] = 2
    params['MAX_DEM_BY_BLOOD']['AB-'] = 3
    params['MAX_DEM_BY_BLOOD']['A-'] = 6
    params['MAX_DEM_BY_BLOOD']['O-'] = 7
    params['MAX_DEM_BY_BLOOD']['A+'] = 14


    params['MAX_DEM_BY_BLOOD']['AB+'] = 0
    params['MAX_DEM_BY_BLOOD']['B+'] = 0
    params['MAX_DEM_BY_BLOOD']['O+'] = 0
    params['MAX_DEM_BY_BLOOD']['B-'] = 0
    params['MAX_DEM_BY_BLOOD']['AB-'] = 0
    params['MAX_DEM_BY_BLOOD']['A-'] = 10
    params['MAX_DEM_BY_BLOOD']['O-'] = 10
    params['MAX_DEM_BY_BLOOD']['A+'] = 0


    params['MAX_DEM_BY_BLOOD']['AB+'] = 3
    params['MAX_DEM_BY_BLOOD']['B+'] = 9
    params['MAX_DEM_BY_BLOOD']['O+'] = 18
    params['MAX_DEM_BY_BLOOD']['B-'] = 2
    params['MAX_DEM_BY_BLOOD']['AB-'] = 3
    params['MAX_DEM_BY_BLOOD']['A-'] = 6
    params['MAX_DEM_BY_BLOOD']['O-'] = 7
    params['MAX_DEM_BY_BLOOD']['A+'] = 14

    #params['DEFAULT_VALUE_DIST']

    # Set here donation by blood type (for blood types that are different than the params['DEFAULT_VALUE_DIST'])
    params['MAX_DON_BY_BLOOD']['AB+'] = 0
    params['MAX_DON_BY_BLOOD']['B+'] = 0
    params['MAX_DON_BY_BLOOD']['O+'] = 0
    params['MAX_DON_BY_BLOOD']['B-'] = 0
    params['MAX_DON_BY_BLOOD']['AB-'] = 0
    params['MAX_DON_BY_BLOOD']['A-'] = 10
    params['MAX_DON_BY_BLOOD']['O-'] = 10
    params['MAX_DON_BY_BLOOD']['A+'] = 0

    params['MAX_DON_BY_BLOOD']['AB+'] = 3
    params['MAX_DON_BY_BLOOD']['B+'] = 9
    params['MAX_DON_BY_BLOOD']['O+'] = 18
    params['MAX_DON_BY_BLOOD']['B-'] = 2
    params['MAX_DON_BY_BLOOD']['AB-'] = 3
    params['MAX_DON_BY_BLOOD']['A-'] = 6
    params['MAX_DON_BY_BLOOD']['O-'] = 7
    params['MAX_DON_BY_BLOOD']['A+'] = 14


    #The default weights to split the demand of a blood type is equal weights. The only requirement is that each
    #weight is positive and they add up to 1. 
    #Default
    params['SURGERYTYPES_PROP'] = {k:1/len(params['Surgerytypes']) for k in params['Surgerytypes']}
    params['SUBSTITUTION_PROP'] = {k:1/len(params['Substitution']) for k in params['Substitution']}

    # Set here the weights for each surgery type (if different than the default)
    params['SURGERYTYPES_PROP']['Urgent'] = 1/2
    params['SURGERYTYPES_PROP']['Elective'] = 1 - params['SURGERYTYPES_PROP']['Urgent']

    # Set here the weights for each substitution type (if different than the default)
    params['SUBSTITUTION_PROP'][True] = 1
    #params['SUBSTITUTION_PROP'][False] = 1 - params['SUBSTITUTION_PROP'][True]


    #Set here random surge parameters
    #params['TIME_PERIODS_SURGE'] = set([4,8,10,12,14])
    params['TIME_PERIODS_SURGE'] = set([3,6,10,13])
    #SURGE_PROB = 0.7
    params['SURGE_FACTOR'] = 6 #The surge demand is always going to be poisson with mean SURGE_FACTOR*params['MAX_DEM_BY_BLOOD'], even if the regular demand distribution is Uniform

    
    #Set here the weights for the utility function - urgent coverage, elective coverage, proportion of blood discarded
    params['WEIGHT_URGENT']=10
    params['WEIGHT_ELECTIVE']=1
    params['WEIGHT_DISCARDED']=3



    if (params['SAMPLING_DIST'] == 'P'):
        params['MAX_DEM_BY_BLOOD'] = {k: int(v * params['POISSON_FACTOR']) for k, v in params['MAX_DEM_BY_BLOOD'].items()}
        params['MAX_DON_BY_BLOOD'] = {k: int(v * params['POISSON_FACTOR']) for k, v in params['MAX_DON_BY_BLOOD'].items()}
        
        params['AVG_TOTAL_DEMAND'] = sum(params['MAX_DEM_BY_BLOOD'].values())
        params['AVG_TOTAL_SUPPLY'] = sum(params['MAX_DON_BY_BLOOD'].values())
        params['NUM_PARALLEL_LINKS'] = int(params['MAX_AGE']/2 * max(params['MAX_DON_BY_BLOOD'].values()))
        #print("Exogenous info dist: Poisson ")
    else:
        params['AVG_TOTAL_DEMAND'] = sum(params['MAX_DEM_BY_BLOOD'].values())/2
        params['AVG_TOTAL_SUPPLY'] = sum(params['MAX_DON_BY_BLOOD'].values())/2
        params['NUM_PARALLEL_LINKS'] = int(params['MAX_AGE']/2 * max(params['MAX_DON_BY_BLOOD'].values()))
        #print("Exogenous info dist: Uniform")

    #Checking if MYOPIC policy
    if not params['USE_VFA']:
        params['ALPHA'] = 0
        params['LOAD_VFA'] = False
        params['SAVE_VFA'] = False
        params['NUM_TRAINNING_ITER'] = 0
        params['NUM_ITER'] = params['NUM_TESTING_ITER']
        params['NUM_PARALLEL_LINKS'] = 1    


    print("Printing params dict\n")
    printParams(params)


    if (params['SAMPLING_DIST'] == 'P'):
         print("Exogenous info dist: Poisson ")
    else:
        print("Exogenous info dist: Uniform")
    
    print("Demand parameters by blood type ",params['MAX_DEM_BY_BLOOD'])
    print("There are ",params['NUM_SUR_TYPES'] * len(params['Substitution'])," demand nodes for each blood type")
    print("Weights SURGERYTYPES_PROP ",params['SURGERYTYPES_PROP'])
    print("Weights SUBSTITUTION_PROP ",params['SUBSTITUTION_PROP'])

    print("Donation parameters by blood type ",params['MAX_DON_BY_BLOOD'])    

    print("AVG TOTAL DEMAND ",params['AVG_TOTAL_DEMAND'])
    print("AVG TOTAL SUPPLY ",params['AVG_TOTAL_SUPPLY'])
    print("NUM PARALLEL LINKS ",params['NUM_PARALLEL_LINKS'])


    print("Possible surge time periods ", params['TIME_PERIODS_SURGE'])
    print("SURGE_PROB ", params['SURGE_PROB'], " and SURGE_FACTOR ", params['SURGE_FACTOR'])
        
        
    return params 





def initOutputListHeaders(params):
    labelsDemandExo=['Iteration','Time','Bloodtype','Urgency','isSubAllowed','DemandValue']
    labelsDonationExo=['Iteration','Time','Bloodtype','DonationValue']
    labelsSupplyPre=['Iteration','Time','BloodType','Age','PreInv']
    labelsSupplyPost=['Iteration','Time','BloodType','Age','PostInv']

    
    labelsSlopesList=['Iteration','Time','BloodType','Age']
    vNames = ["v_"+str(r) for r in list(range(params['NUM_PARALLEL_LINKS']))]
    labelsSlopesList = labelsSlopesList + vNames
   
    headerSolDemList =['Iteration',  'Time','BloodTypeS', 'Age','BloodTypeD', 'Urgency', 'SubsAllowed', 'isCompatible',  'Contrib', 'Value']    
    headerSolHoldList = ['Iteration','Time','BloodTypeS','Age','Value']  
    headerSimuList = ['Iteration','ElapsedTime','Stepsize','ObjVal','isTrainning']  
    headerUpdateVfaList = ['Iteration','Time','BloodType','Age','R','vhat','vbarOld','sqGrad','stepsize','vbarNew']
    
    return(labelsDemandExo, labelsDonationExo,labelsSupplyPre,labelsSupplyPost,labelsSlopesList,headerSolDemList,headerSolHoldList,headerSimuList,headerUpdateVfaList)


def convertToDfOutputLists(params,Bld_Net,demandExoList, donationExoList, supplyPreList, supplyPostList, slopesList, solDemList, solHoldList,  simuList, updateVfaList):
    labelsDemandExo, labelsDonationExo, labelsSupplyPre,  labelsSupplyPost, labelsSlopesList, headerSolDemList, headerSolHoldList, headerSimuList, headerUpdateVfaList = initOutputListHeaders(params)

    #Flatteting the lists 
    dfSimu = pd.DataFrame.from_records(simuList,columns=headerSimuList)

    demandExoListFlat = [(ite,t,dnode[0],dnode[1],dnode[2],dvalue) for ite,t,d in demandExoList for dnode,dvalue in zip(Bld_Net.demandnodes,d)]
    dfDemandExo = pd.DataFrame.from_records(demandExoListFlat,columns=labelsDemandExo)

    donationExoListFlat = [(ite,t,dtype,dvalue) for ite,t,d in donationExoList for dtype,dvalue in zip(params['Bloodtypes'],d)]
    dfDonationExo = pd.DataFrame.from_records(donationExoListFlat,columns=labelsDonationExo)

    supplyPreListFlat = [(ite,t,bnode[0],bnode[1],bvalue) for ite,t,b in supplyPreList for bnode,bvalue in zip(Bld_Net.bloodnodes,b)]
    dfSupplyPre = pd.DataFrame.from_records(supplyPreListFlat,columns=labelsSupplyPre)


    supplyPostListFlat = [(ite,t,bnode[0],bnode[1],bvalue) for ite,t,b in supplyPostList for bnode,bvalue in zip(Bld_Net.bloodnodes,b)]
    dfSupplyPost = pd.DataFrame.from_records(supplyPostListFlat,columns=labelsSupplyPost)


    solDemListFlat = [(ite,t,bld[0],bld[1],dem[0],dem[1],dem[2],params['SubMatrix'][(bld[0], dem[0])],Bld_Net.demweights[(bld,dem)],xbd) for ite,t,xDem in solDemList for bld,xb in zip(Bld_Net.bloodnodes,xDem) for dem,xbd in zip(Bld_Net.demandnodes,xb)]
    dfSolDem = pd.DataFrame.from_records(solDemListFlat,columns=headerSolDemList)

    solHoldListFlat = [(ite,t,bnode[0],bnode[1],hvalue) for ite,t,h in solHoldList for bnode,hvalue in zip(Bld_Net.bloodnodes,h)]
    dfSolHold = pd.DataFrame.from_records(solHoldListFlat,columns=headerSolHoldList)


    slopesListFlat = [(vnode[0],vnode[1],bnode[0],bnode[1],*list(vnode[2])) for vnode,bnode in zip(slopesList,Bld_Net.bloodnodes*params['NUM_ITER']*params['MAX_TIME'])]

    dfSlopes = pd.DataFrame.from_records(slopesListFlat,columns=labelsSlopesList)

    dfUpdateVfa = pd.DataFrame.from_records(updateVfaList,columns=headerUpdateVfaList)

    return dfDemandExo, dfDonationExo, dfSupplyPre, dfSupplyPost, dfSlopes, dfSolDem, dfSolHold,  dfSimu, dfUpdateVfa

def printDfsToOutputFile(params,dfDemandExo, dfDonationExo, dfSupplyPre, dfSupplyPost, dfSlopes, dfSolDem, dfSolHold,  dfSimu, dfUpdateVfa):

    t_init_print = time.time()
    print("Started printing file")
    # print to excel file
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(params['OUTPUT_FILENAME'], engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    dfSimu.to_excel(writer, sheet_name='Simu')
    
    if params['PRINT_ALL']:
        dfDemandExo.to_excel(writer, sheet_name='DemandExo')
        dfDonationExo.to_excel(writer, sheet_name='DonationExo')
        dfSupplyPre.to_excel(writer, sheet_name='SupplyPre')
        dfSolDem.to_excel(writer, sheet_name='SolDem')
        dfSolHold.to_excel(writer, sheet_name='HoldDem')
        dfSupplyPost.to_excel(writer, sheet_name='SupplyPost')
        dfSlopes.to_excel(writer, sheet_name='SlopesList')
        dfUpdateVfa.to_excel(writer, sheet_name='UpdatesVfa')


    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    print("Finished printing files in {:.2f} secs".format(time.time()-t_init_print))
  
   
def Main():    
    
    t_global_init = time.time()
    print("********************Started Main*****************\n")
    params = loadParams('Parameters.xlsx')
    alpha = params['ALPHA']
    
    #ite_TRA=np.arange(0, params['NUM_TRAINNING_ITER'], 1)
    #selectedIte = list(set([0,5,10,19]) & set(ite_TRA))
    
    

    # initializing the random seed for trainning iterations
    np.random.seed(params['SEED_TRAINING'])

    # initializing the blood network
    Bld_Net = create_bld_net(params)

    if (params['LOAD_VFA'] and os.path.exists(params['NAME_LOAD_VFA_PICKLE'])):
        pickle_off = open(params['NAME_LOAD_VFA_PICKLE'],"rb")
        Other_Bld_Net = pickle.load(pickle_off)
        Bld_Net.varr = Other_Bld_Net.varr.copy()
        Bld_Net.parallelarr = Other_Bld_Net.parallelarr.copy()

    # initializing the model
    state_names = ['BloodInventory', 'Demand', 'Donation']
    decision_names = ['Hold', 'Contribution']  

    # initializing the lists that will store the all the info/decisions/states/slopes along the iterations for printing purposes
    demandExoList, donationExoList, supplyPreList, supplyPostList, slopesList, solDemList, solHoldList,  simuList, updateVfaList = [],[],[],[],[],[],[],[],[]

    #initializing the policy
    P = Policy(params,Bld_Net)

    
    iteration = 0
    obj = []

    if (params['NUM_TRAINNING_ITER']>0):
        print("\n Starting training iterations\n")

    while iteration < params['NUM_ITER']:  
        IS_TRAINING = (iteration<params['NUM_TRAINNING_ITER'])
        if (iteration==params['NUM_TRAINNING_ITER']):
            print("Starting testing iterations! Currently at iteration ",iteration)
            print("Reseting random seed!")
            np.random.seed(params['SEED_TESTING'])
            
        t_init = time.clock()
        print('Iteration = ', iteration)
        
        # Initial inventory
        if (params['SAMPLING_DIST'] == 'P'):
            bldinv_init = [int(np.random.poisson(params['MAX_DON_BY_BLOOD'][bld[0]])*.9) if bld[1]=='0' else int(np.random.poisson(params['MAX_DON_BY_BLOOD'][bld[0]])*(0.1/(params['MAX_AGE']-1))) for bld in Bld_Net.bloodnodes]
        else:
            bldinv_init = [round(np.random.uniform(0, params['MAX_DON_BY_BLOOD'][bld[0]])*.9) if bld[1]=='0' else round(np.random.uniform(0, params['MAX_DON_BY_BLOOD'][bld[0]])*(0.1/(params['MAX_AGE']-1))) for bld in Bld_Net.bloodnodes]
    
        # Initial exogenous information
        if (params['SAMPLING_DIST'] == 'P'):
            exog_info_init = generate_exog_info_by_bloodtype_p(0, Bld_Net, params)
        else:
            exog_info_init = generate_exog_info_by_bloodtype(0, Bld_Net, params)

        #initial state - the donation is irrelevant at time period zero - only the initial invetory counts
        init_state = {'BloodInventory': bldinv_init, 'Demand': exog_info_init.demand, 'Donation' : exog_info_init.donation}

        M = Model(state_names, decision_names, init_state, Bld_Net,params)
        #print("Initial blood supply across {} types and {} ages is {}".format(params['NUM_BLD_TYPES'],params['MAX_AGE'],sum(M.bld_inv)))
        #print("Initial demand across {} types and {} urgency states and {} substitution states is {}".format(params['NUM_BLD_TYPES'],params['NUM_SUR_TYPES'],len(params['Substitution']),sum(M.demand)))

        
        t = 0        
        obj.append(0)

        #Steping forward in time
        while t < params['MAX_TIME']:
          
            
            #Compute the solution for time period t - return the solution, the value, the dual and the updated lists
            sol,val,x,hld,d,solDemList,solHoldList=P.getLPSol(params,M,iteration,t,solDemList,solHoldList,IS_TRAINING)
            obj[iteration] += val
            
            
            #Grabbing exogenous data to construct data frame
            recordDemandExo = (iteration,t,M.Bld_Net.demandamount.copy())
            demandExoList.append(recordDemandExo)       
            if (t==0):   
                recordDonationExo = (iteration,0,list(np.array(M.bld_inv)[::params['MAX_AGE']]))
                donationExoList.append(recordDonationExo)
            
            if (t<params['MAX_TIME']-1):
                recordDonationExo = (iteration,t+1,M.donation.copy())
                donationExoList.append(recordDonationExo)
                
            #Grabbing pre-decision state to construct data frame
            recordSupplyPre = (iteration,t,M.bld_inv.copy())
            supplyPreList.append(recordSupplyPre)
                
            
            if IS_TRAINING:    
                alpha,slopesList,updateVfaList = P.updateVFAs(params,M,iteration,t,d, slopesList,updateVfaList)
                            
            # build decision
            dcsn = M.build_decision({'Hold': hld, 'Contribution': val})
                    
            M.transition_fn(dcsn)
            
            #Grabbing post-decision state to construct data frame
            recordSupplyPost = (iteration,t,M.bld_inv.copy())
            supplyPostList.append(recordSupplyPost)
            
            t += 1
            # generate/read exogenous information 
            if (params['SAMPLING_DIST'] == 'P'):
                exog_info = generate_exog_info_by_bloodtype_p(t, Bld_Net, params)
            else:
                exog_info = generate_exog_info_by_bloodtype(t, Bld_Net, params)
            M.exog_info_fn(exog_info)
            
        
        
        # copy v to the parallel links
        for t in params['Times']:
            for hld in M.Bld_Net.holdnodes:
                parArr = 1 * M.Bld_Net.varr[(t,hld, M.Bld_Net.supersink)]
                M.Bld_Net.add_parallel(t,hld, M.Bld_Net.supersink, parArr)
        
          

        t_end = time.clock()
        recordSimu = (iteration,int(t_end-t_init),alpha,obj[iteration],(iteration<params['NUM_TRAINNING_ITER']))
        simuList.append(recordSimu)
       
        print("***Finishing iteration {} in {:.2f} secs. Total contribution: {:.2f}***\n".format(recordSimu[0],recordSimu[1],recordSimu[3]))
        
        
        iteration += 1
        
    #End of iterations
    ###########################################################################################################################################


    if (params['SAVE_VFA']):
        pickling_on = open(params['NAME_SAVE_VFA_PICKLE'],"wb")
        pickle.dump(M.Bld_Net, pickling_on)
        pickling_on.close()

    

    ###########################################################################################################################################
    #Computing stats and plots
    ###########################################################################################################################################
    dfDemandExo, dfDonationExo, dfSupplyPre, dfSupplyPost, dfSlopes, dfSolDem, dfSolHold,  dfSimu, dfUpdateVfa = convertToDfOutputLists(params,Bld_Net,demandExoList, donationExoList, supplyPreList, supplyPostList, slopesList, solDemList, solHoldList,  simuList, updateVfaList)
        

    policy = params['USE_VFA'] and 'VFA-Based' or 'MYOPIC'
    surge  = params['SURGE_PROB']>0 and "SURGE_"+str(params['SURGE_PROB']) or "NO_SURGE"
    instance = "Policy{}_{}_PEN_{:,}_ALPHA_{:.2f}".format(policy,surge,params['BLOOD_FOR_ELECTIVE_PENALTY'],params['ALPHA'])

    #Average Contribution
    meanTesting = dfSimu.groupby('isTrainning')['ObjVal'].mean()[False]

    #Total Blood discarded
    totalDiscarded = dfSolHold.loc[(dfSolHold.Age.astype(int)>params['MAX_AGE']-2)&(dfSolHold.Iteration>=params['NUM_TRAINNING_ITER']),:].copy()['Value'].sum()

    #Total Donation
    totalDonation = dfDonationExo[dfDonationExo.Iteration>=params['NUM_TRAINNING_ITER']].copy()['DonationValue'].sum()

    #Coverage 
    dfCoverage = dfSolDem.groupby(['BloodTypeD', 'Urgency','Iteration',  'Time'])['Value'].sum()
    dfCoverage.index = dfCoverage.index.rename("Bloodtype", level=0)   
    dfCoverage = pd.concat([dfCoverage,dfDemandExo.groupby(['Bloodtype', 'Urgency','Iteration',  'Time'])['DemandValue'].sum()],axis=1)
    dfCoverage['Ratio']= dfCoverage['Value']/ dfCoverage['DemandValue']
    
    
    dfCoverage_agg_ite = dfCoverage.groupby(['Bloodtype', 'Urgency','Iteration'])['Ratio'].mean().reset_index()

    numTra = params['NUM_TRAINNING_ITER']
    dfCoverage_agg_test = dfCoverage_agg_ite.query('Iteration >= @numTra')
    dfPrintIte=dfCoverage_agg_test.pivot_table('Ratio',index='Bloodtype',columns='Urgency')

    

    finalCoverage=dfCoverage_agg_test.groupby('Urgency')['Ratio'].mean()

    coverage = "Average Coverage: -  Urgent: {:.2f} Elective: {:.2f} Avg: {:.2f}".format(finalCoverage['Urgent'],finalCoverage['Elective'],dfCoverage_agg_ite['Ratio'].mean())
    
    #dfUtility = dfCoverage.query('Iteration >= @numTra').copy().reset_index()
    #dfUtility['Weight']=-1
    #dfUtility['Score']=0
    #dfUtility.loc[dfUtility.Urgency=="Elective",'Weight']=1
    #dfUtility.loc[dfUtility.Urgency=="Urgent",'Weight']=100
    #dfUtility.loc[(dfUtility.Urgency=="Urgent") & (dfUtility.Ratio>.9),'Score']=1
    #sumRatio=(dfUtility['Ratio']*dfUtility['Weight']).sum()
    #sumWeight=(dfUtility['Weight']).sum()
    #sumScore=(dfUtility['Score']).sum()
    #utility=sumRatio/sumWeight
    

    #Utility function
    utility=(params['WEIGHT_URGENT']*round(finalCoverage['Urgent'],2)+params['WEIGHT_ELECTIVE']*round(finalCoverage['Elective'],2))*100
    modifiedUtil=utility-params['WEIGHT_DISCARDED']*100*round(totalDiscarded/totalDonation,2)



    ###########################################################################################################################################
    #Figure 1 - Total Contribution along iterations
    ite = np.arange(0, params['NUM_ITER'], 1)
    ite_TRA =  np.arange(0, params['NUM_TRAINNING_ITER'], 1)
    ite_TES =  np.arange(0, params['NUM_TESTING_ITER'], 1) + params['NUM_TRAINNING_ITER']
    
    fig_ite, ax_ite = plt.subplots(figsize=(16,8))
    ax_ite.plot(ite,dfSimu['ObjVal'],'g-',label='_nolegend_')
    ax_ite.plot(ite_TRA,dfSimu['ObjVal'][ite_TRA],'g-',label="Training",marker='o')
    ax_ite.plot(ite_TES,dfSimu['ObjVal'][ite_TES],'b-',label="Testing",marker='o')
    ax_ite.hlines(meanTesting, ite_TES[0], ite_TES[-1], color='b',linestyle='--',label="Avg Testing",linewidth=4)
    
    ax_ite.axvline(ite_TES[0], 0, 1, color='k',linestyle=':')
    ax_ite.legend()
    ax_ite.set_xlabel('Iterations',fontsize=12)
    ax_ite.set_ylabel('$',fontsize=12)
    #ax_ite.set_ylim([20000,34000])
    ax_ite.set_title("Policy {}_{} - Total contributions \n Avg total contribution during TESTING iterations: ${:,}\n Final utility: {:.0f}".format(policy,surge,meanTesting,modifiedUtil))
    ax_ite.set_xticks(ite)
    ax_ite.set_xticklabels(list(ite_TRA)+list(np.arange(0, params['NUM_TESTING_ITER'], 1)))
    for c in ite_TES:
        ax_ite.get_xticklabels()[c].set_color("b")
    ###########################################################################################################################################

    
    ###########################################################################################################################################
    #Figure 2 - Exogenous processes - Demand and Donation
    fig_exo, ax_exo = plt.subplots(2,1,figsize=(16,8),sharex=True)
    dfDemandExoP = dfDemandExo[dfDemandExo.Iteration>=params['NUM_TRAINNING_ITER']].copy()
    dfDemandExoP['Iteration'] = dfDemandExoP['Iteration'] - params['NUM_TRAINNING_ITER']
    dfPrintDemand = dfDemandExoP.pivot_table('DemandValue',index='Time',columns='Iteration',aggfunc='sum')
    l=dfPrintDemand.plot(ax=ax_exo[0],title="Total Demand - {}".format(surge),legend=False)

    dfDonationExoP = dfDonationExo[dfDonationExo.Iteration>=params['NUM_TRAINNING_ITER']].copy()
    dfDonationExoP['Iteration'] = dfDonationExoP['Iteration'] - params['NUM_TRAINNING_ITER']
    dfPrintDonation = dfDonationExoP.pivot_table('DonationValue',index='Time',columns='Iteration',aggfunc='sum')
    dfPrintDonation.plot(ax=ax_exo[1],title="Total Donation",legend=False)
    ax_exo[1].set_xlabel("Time period",fontsize=12)
    ax_exo[0].set_ylabel("Units",fontsize=12)
    ax_exo[1].set_ylabel("Units",fontsize=12)
    
    fig_exo.legend(l,labels=list(np.arange(0, params['NUM_TESTING_ITER'], 1)),title="Iteration",loc="center right",fancybox=True, shadow=True)
    fig_exo.suptitle("Exogenous processes along the testing iterations")
    ###########################################################################################################################################

    
    
    dfInv = dfSupplyPre[dfSupplyPre.Iteration>=params['NUM_TRAINNING_ITER']].copy()
    dfInv['Iteration'] = dfInv['Iteration'] - params['NUM_TRAINNING_ITER']
    
    ###########################################################################################################################################
    #Figure 3 - Pre decision inventory levels by age
    fig_inv, ax_inv = plt.subplots(3,1,figsize=(16,10),sharex=True)
    for age in [0,1,2]:

        strage=str(age)

        dfPrint = dfInv[dfInv.Age == strage].pivot_table('PreInv',index=['Time'],columns=['Iteration'],aggfunc='sum')
        l=dfPrint.plot(ax=ax_inv[age],legend=False)


        dfPrintAvg = dfInv[dfInv.Age == strage].groupby(['Time','Iteration'])['PreInv'].sum()
        dfPrintAvg = dfPrintAvg.groupby('Time').mean()
        avg_line = ax_inv[age].plot(dfPrintAvg.index,dfPrintAvg.values,'k',linestyle=':',marker='s',markersize='12',label='Average')
        ax_inv[age].set_title("Age: {} - Avg inventory level: {:.0f}".format(age,dfPrintAvg.values.mean()))
        ax_inv[age].set_ylabel("Units",fontsize=12)

    ax_inv[2].set_xlabel("Time period",fontsize=12)
    fig_inv.suptitle("Policy {}_{} \n Pre-decision  inventory level all blood types".format(policy,surge))
    fig_inv.legend([l,avg_line],labels=list(np.arange(0, params['NUM_TESTING_ITER'], 1))+["Avg"],title="Iteration",loc='center right',fancybox=True, shadow=True, ncol=1)
    ###########################################################################################################################################


    ###########################################################################################################################################
    #Figure 4 - Pre decision inventory levels by bloodtype
    fig_inv_blood, ax_inv_blood = plt.subplots(4,2,figsize=(16,10),sharex=True)
    row = -1
    for m,b in enumerate(params['Bloodtypes']):

        col= (m)%2
        if col == 0:
            row+=1
            ax_inv_blood[row,col].set_ylabel("Units",fontsize=8)

        dfPrint = dfInv[dfInv.BloodType == b].pivot_table('PreInv',index=['Time'],columns=['Iteration'],aggfunc='sum')
        l=dfPrint.plot(ax=ax_inv_blood[row,col],legend=False)

        dfPrintAvg = dfInv[dfInv.BloodType == b].groupby(['Time','Iteration'])['PreInv'].sum()
        dfPrintAvg = dfPrintAvg.groupby('Time').mean()

        avg_line =  ax_inv_blood[row,col].plot(dfPrintAvg.index,dfPrintAvg.values,'k',linestyle=':',marker='s',markersize='10',label='Average')

        ax_inv_blood[row,col].set_title("Bloodtype: {}  - Avg inventory level: {:.0f}".format(b,dfPrintAvg.values.mean()),fontsize=8)
        ax_inv_blood[row,col].set_xlabel("Time Period",fontsize=8)

    fig_inv_blood.suptitle("Policy {}_{} \n Pre-decision  inventory level all ages".format(policy,surge))
    fig_inv_blood.legend([l,avg_line],labels=list(np.arange(0, params['NUM_TESTING_ITER'], 1))+["Avg"],loc="center right", title="Iteration")
    ###########################################################################################################################################


    ###########################################################################################################################################
    #Figure 5 - Pre decision inventory levels 
    dfPrintAvg = dfInv.groupby(['Time','Iteration'])['PreInv'].sum()
    dfPrintAvg = dfPrintAvg.groupby('Time').mean()
    fig_inv_total, ax_inv_total = plt.subplots(figsize=(16,8))
    dfPrint = dfInv.pivot_table('PreInv',index=['Time'],columns=['Iteration'],aggfunc='sum')
    dfPrint.plot(ax=ax_inv_total,legend=True)
    first_legend = ax_inv_total.legend(title="Iteration")
    avg_line = ax_inv_total.plot(dfPrintAvg.index,dfPrintAvg.values,'k',linestyle=':',marker='s',markersize='12',label='Average')
    ax_inv_total.legend(title="Iteration")
    ax_inv_total.set_xlabel("Time period")
    ax_inv_total.set_ylabel("Units")
    fig_inv_total.suptitle("Policy {}_{} \n Pre-decision inventory level all blood types and ages \n Average inventory level across time periods: {:.0f}".format(policy,surge,dfPrintAvg.values.mean()))
    ###########################################################################################################################################


    
    

    ###########################################################################################################################################
    #Figure 6 - Discarded blood during testing iterations
    dfDiscarded = dfSolHold.loc[(dfSolHold.Age.astype(int)>params['MAX_AGE']-2)&(dfSolHold.Iteration>=params['NUM_TRAINNING_ITER']),:].copy()
    dfDiscarded = dfDiscarded.groupby(['BloodTypeS','Time'])['Value'].sum().reset_index()
    dfDiscarded['Prop']=100*dfDiscarded['Value']/totalDonation
    

    y_a = dfDiscarded.BloodTypeS.unique()
    x_a = dfDiscarded.Time.unique()
    discarded_matrix = np.reshape(np.array(dfDiscarded['Value']), (-1, len(x_a)))
   
    
    fig_dis, ax_dis = plt.subplots(figsize=(16,8))  
    im = ax_dis.imshow(discarded_matrix, cmap='hot_r',origin='lower',aspect='auto',alpha=.9)
    cbar = ax_dis.figure.colorbar(im, ax=ax_dis,label='Units of blood')  
    ax_dis.set_xticks(np.arange(len(x_a)))
    ax_dis.set_yticks(np.arange(len(y_a)))
    ax_dis.set_xticklabels(x_a)
    ax_dis.set_yticklabels(y_a)
    ax_dis.set_xlabel("Time Period")
    ax_dis.set_title("Policy {}_{} \n Total discarded blood during TESTING iterations\n Proportion of blood discarded: {:.2f}%".format(policy,surge,totalDiscarded*100/totalDonation))

    ###########################################################################################################################################


    ###########################################################################################################################################
    #Figure 7 - Demand Coverage - Testing iterations
    fig_tes, ax_tes = plt.subplots(figsize=(16,8))   
    ax_tes.plot(dfPrintIte['Urgent'],marker='o')
    ax_tes.plot(dfPrintIte['Elective'],marker='o')
    ax_tes.set_title("Policy {}_{} \n Average coverage of demand by blood type and urgency level during TESTING iterations\n {} Coverage utility: {:.0f}".format(policy,surge,coverage,utility))
    ax_tes.set_ylabel("Coverage ratio")
    ax_tes.legend(title="Urgency level",loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True, ncol=1)
    ###########################################################################################################################################


    ###########################################################################################################################################
    #Figure 8 - Demand Coverage - Along Blood types
    iteS = np.arange(0, params['NUM_ITER'], 1)
    if True:
        
        #selectedIte = sorted(list(set([0,5,10,19]) ))
        selectedIte = sorted(list(set([0,5,10,19]) & set(iteS)))
        fig_cover, ax_cover = plt.subplots(1,len(selectedIte),figsize=(16,8),sharey=True)

        

        for i,ite in enumerate(selectedIte):
            dfPrintIte = dfCoverage_agg_ite[dfCoverage_agg_ite.Iteration==ite]
            dfPrintIte=dfPrintIte.pivot_table('Ratio',index='Bloodtype',columns='Urgency')

            typIte='TES'
            if ite <params['NUM_TRAINNING_ITER']:
                typIte='TRA'

            
            if len(selectedIte)==1:
                ax_cover.plot(dfPrintIte['Urgent'],marker='o')
                ax_cover.plot(dfPrintIte['Elective'],marker='o')
                ax_cover.set_title("Iteration {} ({})".format(ite,typIte))
            else:
                ax_cover[i].plot(dfPrintIte['Urgent'],marker='o')
                ax_cover[i].plot(dfPrintIte['Elective'],marker='o')
                ax_cover[i].set_title("Iteration {} ({})".format(ite,typIte))
           
            fig_cover.suptitle("Policy {}-{} \n Average coverage of demand by blood type and urgency level for different iterations".format(policy,surge))

        if len(selectedIte)==1:
            ax_cover.set_ylabel("Coverage ratio")
            ax_cover.legend(title="Urgency level",loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True, ncol=1)
        else:
            ax_cover[0].set_ylabel("Coverage ratio")
            ax_cover[len(selectedIte)-1].legend(title="Urgency level",loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True, ncol=1)    
    ###########################################################################################################################################


    ###########################################################################################################################################
    #Figure 9 - Demand Coverage - Along time periods
    if True:
        dfCoverage_agg_ite = dfCoverage.groupby(['Time', 'Urgency','Iteration'])['Ratio'].mean().reset_index()
        #selectedIte = sorted(list(set([0,5,10,19]) ))
        selectedIte = sorted(list(set([0,5,10,19]) & set(iteS)))
        fig_cover_ite, ax_cover_ite = plt.subplots(1,len(selectedIte),figsize=(16,8),sharey=True,sharex=True)
        for i,ite in enumerate(selectedIte):
            dfPrintIte = dfCoverage_agg_ite[dfCoverage_agg_ite.Iteration==ite]
            dfPrintIte=dfPrintIte.pivot_table('Ratio',index='Time',columns='Urgency')

            typIte='TES'
            if ite <params['NUM_TRAINNING_ITER']:
                typIte='TRA'
            
            if len(selectedIte)==1:
                ax_cover_ite.plot(dfPrintIte['Urgent'],marker='o')
                ax_cover_ite.plot(dfPrintIte['Elective'],marker='o')
                ax_cover_ite.set_title("Iteration {} ({})".format(ite,typIte))
                ax_cover_ite.set_xticks(list(range(0,params['MAX_TIME'],2)))
                ax_cover_ite.set_xticklabels(list(range(0,params['MAX_TIME'],2)))
                ax_cover_ite.set_xlabel("Time Period")
            else:
                ax_cover_ite[i].plot(dfPrintIte['Urgent'],marker='o')
                ax_cover_ite[i].plot(dfPrintIte['Elective'],marker='o')
                ax_cover_ite[i].set_title("Iteration {} ({})".format(ite,typIte))
                ax_cover_ite[i].set_xticks(list(range(0,params['MAX_TIME'],2)))
                ax_cover_ite[i].set_xticklabels(list(range(0,params['MAX_TIME'],2)))
                ax_cover_ite[i].set_xlabel("Time Period")

            fig_cover_ite.suptitle("Policy {}-{} \n Average coverage of demand by time period and urgency level for different iterations".format(policy,surge))

        if len(selectedIte)==1:
            ax_cover_ite.set_ylabel("Coverage ratio")
            ax_cover_ite.legend(title="Urgency level",loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True, ncol=1)
        else:
            ax_cover_ite[0].set_ylabel("Coverage ratio")
            ax_cover_ite[len(selectedIte)-1].legend(title="Urgency level",loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True, ncol=1)


    ###########################################################################################################################################
    #Figure 10 - Histogram Demand Coverage - Along time periods       
    idx = pd.IndexSlice
    dfCoverage=dfCoverage[dfCoverage.DemandValue>0].copy()

    uncoverU = dfCoverage.loc[idx[:,['Urgent'],ite_TES,:],['DemandValue']].values.sum() - dfCoverage.loc[idx[:,['Urgent'],ite_TES,:],['Value']].values.sum() 
    uncoverE = dfCoverage.loc[idx[:,['Elective'],ite_TES,:],['DemandValue']].values.sum() - dfCoverage.loc[idx[:,['Elective'],ite_TES,:],['Value']].values.sum() 
    uncoverT = dfCoverage.loc[idx[:,:,ite_TES,:],['DemandValue']].values.sum() - dfCoverage.loc[idx[:,:,ite_TES,:],['Value']].values.sum() 

    demandU = dfCoverage.loc[idx[:,['Urgent'],ite_TES,:],['DemandValue']].values.sum() 
    demandE = dfCoverage.loc[idx[:,['Elective'],ite_TES,:],['DemandValue']].values.sum() 
    demandT = dfCoverage.loc[idx[:,:,ite_TES,:],['DemandValue']].values.sum() 
    

    fig_hist, ax_hist = plt.subplots(1,3,figsize=(16,8),sharey=True,sharex=True)
    ax_hist[0].hist(dfCoverage.loc[idx[:,['Urgent'],ite_TES,:],['Ratio']].values, bins=11,color='tab:blue')
    ax_hist[0].set_title("Urgent")
    ax_hist[0].set_ylabel("Count")
    ax_hist[0].set_xlim([0,1])
    ax_hist[0].annotate('Uncovered Demand: {:,}\n Total Demand: {:,}'.format(int(uncoverU),demandU),xy=(.8, .975), xycoords='axes fraction',horizontalalignment='right', verticalalignment='top',fontsize=12)
    
    ax_hist[1].hist(dfCoverage.loc[idx[:,['Elective'],ite_TES,:],['Ratio']].values, bins=11,color='tab:orange')
    ax_hist[1].set_title("Elective")
    ax_hist[1].set_xlabel("Coverage Ratio")
    ax_hist[1].annotate('Uncovered Demand: {:,}\n Total Demand: {:,}'.format(int(uncoverE),demandE),xy=(.8, .975), xycoords='axes fraction',horizontalalignment='right', verticalalignment='top',fontsize=12)


    ax_hist[2].hist(dfCoverage.loc[idx[:,:,ite_TES,:],['Ratio']].values, bins=11,color='tab:gray')
    ax_hist[2].set_title("Total")
    ax_hist[2].annotate('Uncovered Demand: {:,}\n Total Demand: {:,}'.format(int(uncoverT),demandT),xy=(.8, .975), xycoords='axes fraction',horizontalalignment='right', verticalalignment='top',fontsize=12)

    fig_hist.suptitle("Policy {}-{} \n Histogram of blood coverage - All TESTING iterations and time periods\n {} Coverage utility: {:.0f}".format(policy,surge,coverage,utility))
    ###########################################################################################################################################
    
    if params['SAVE_PLOTS']:
        fig_ite.savefig('{}_Figure1.pdf'.format(instance))
        fig_exo.savefig('{}_Figure2.pdf'.format(instance))
        fig_inv.savefig('{}_Figure3.pdf'.format(instance))
        fig_inv_blood.savefig('{}_Figure4.pdf'.format(instance))
        fig_inv_total.savefig('{}_Figure5.pdf'.format(instance))
        fig_dis.savefig('{}_Figure6.pdf'.format(instance))
        fig_tes.savefig('{}_Figure7.pdf'.format(instance))
        fig_cover.savefig('{}_Figure8.pdf'.format(instance))
        fig_cover_ite.savefig('{}_Figure9.pdf'.format(instance))
        fig_hist.savefig('{}_Figure10.pdf'.format(instance))

    if params['SHOW_PLOTS']:
        plt.show()

    ###########################################################################################################################################
    #Printing the final results
    print("\n*******************************************************************************************")
    print("Policy {}_{}".format(policy,surge))
    print(instance)
    print("Average total contribution during TESTING iterations: ${:,}".format(meanTesting))
    print(coverage)
    print("Proportion of blood discarded: {:.2f}% ".format(totalDiscarded*100/totalDonation))
    print("Final utility: {:.0f}".format(modifiedUtil))
    print("*********************************************************************************************\n")


    with open("OutputAll.txt", "a") as myfile:
        print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(instance,meanTesting,finalCoverage['Urgent'],finalCoverage['Elective'],dfCoverage_agg_ite['Ratio'].mean(),totalDiscarded/totalDonation,modifiedUtil),file=myfile)

    ###########################################################################################################################################
  
    print("Total elapsed time {:.2f} secs".format(time.time()- t_global_init))

    ###########################################################################################################################################
    #Printing output file
    if params['PRINT']:
        printDfsToOutputFile(params,dfDemandExo, dfDonationExo, dfSupplyPre, dfSupplyPost, dfSlopes, dfSolDem, dfSolHold,  dfCoverage, dfUpdateVfa)
    ###########################################################################################################################################


    

#End Main
###############################################################################################################################################
                                 






###############################################################################################################################################
if __name__ == "__main__":
    Main()
    
    #mem = max(memory_usage(proc=Main))
    #print("Maximum memory used: {0} MiB".format(str(mem)))
    
    
