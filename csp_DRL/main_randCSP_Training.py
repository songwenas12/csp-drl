import sys
sys.path.append('../or-tools-6.7.2-customized/ortools/gen')

import csv, datetime, glob, os, shutil, time
from random import randint

import RBGenerator
import solverConstructor_RBGenerator
from ortools.constraint_solver import pywrapcp

# Config
training_pool_size = 1000
testing_pool_size = 200
warmup_instance_num = 0

# Pretrained model
use_pre_Train = False
pretrained_model_name = '2_15'
pretrained_model_path = os.path.join('./pre_trained_models', pretrained_model_name)

# CP solver config
solverConstructor_RBGenerator.useTimeLimit = False
solverConstructor_RBGenerator.solverNodeLimit = 1 * 10000
solverConstructor_RBGenerator.solverTimeLimit = 100 * 1000

# RBGenerator config
ifSat = True
k = 2  # Arity
n = 15  # nVar
alpha = 0.7  # domain scale
r = 3  # nConstr
p = 0.21  # tightness

# k = 3  # Arity
# n = 10  # nVar
# alpha = 0.7  # domain scale
# r = 2.5  # nConstr
# p = 0.24  # tightness


def solveInstance(solver, varList):
    tmp_t0 = time.time()
    solver.NextSolution_DRL()
    for var in varList:
        print(var, end=" ")
    print('');
    solver.EndSearch()
    tmp_t1 = time.time();
    return tmp_t1-tmp_t0

def test(masterDRL, iter, test_result_file_path):
    # Recorders
    time_DQN = 0.0
    nTimeOut_DQN = 0.0
    nDecisions_DQN = 0.0
    nBranches_DQN = 0.0
    nFails_DQN = 0.0

    for i in range(testing_pool_size):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"-----Python: testing round "+str(i)+"-----");
        # Solve using the DQN
        testCasePath=testing_pool_path+str(i)+".txt";
        solver, varList, timeLimit = solverConstructor_RBGenerator.constructSolver_Testing(i, testCasePath, masterDRL);
        time=solveInstance(solver, varList);
        time_DQN += solver.getRealSolvingTime()
        nTimeOut_DQN += timeLimit.Crossed()
        nDecisions_DQN += solver.getNumDecisions()
        nBranches_DQN += solver.Branches()
        nFails_DQN += solver.Failures()
        print("[DQN] Time used=" + str(solver.getRealSolvingTime())+", timeOut=" + str(timeLimit.Crossed()) +
              ", nDecisions=" + str(solver.getNumDecisions()) +
              ", nBranches=" + str(solver.Branches())
              +", nFails=" + str(solver.Failures()) + "\n");

    # Record the average test result
    time_DQN /= testing_pool_size
    nDecisions_DQN /= testing_pool_size
    nBranches_DQN /= testing_pool_size
    nFails_DQN /= testing_pool_size
    with open(test_result_file_path, 'a', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerow([iter, time_DQN, nTimeOut_DQN, nDecisions_DQN, nBranches_DQN, nFails_DQN])

    print ("Test done. avgTime[DQN]="+str(time_DQN));
    return time_DQN


# Initialize the result file
def initResultFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)    # Remove the existing one
    # Create a new one with header
    headerStr = ['Iter', 'Time_DQN', 'nTimeOut_DQN', 'nDecision_DQN', 'nBranches_DQN', 'nFails_DQN']
    with open(filePath, 'a', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerow(headerStr)


if __name__=='__main__':

    # Create the log folder
    logFolder = os.path.join("./log", 'train_log_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(logFolder):
        os.makedirs(logFolder)
    with open(os.path.join(logFolder, 'RBConfig.txt'), 'w') as RBConfigFile:
        RBConfigFile.write("k\tn\talpha\tr\tp\n")
        RBConfigFile.write(str(k) + "\t" + str(n) + "\t" + str(alpha) + "\t" + str(r) + "\t" + str(p) + "\n")
    shutil.copy("drlConfig.cfg", os.path.join(logFolder, 'drlConfig.cfg'))  # Copy the drlConfig

    # Initialization
    masterDRL = pywrapcp.Solver.createMasterDRL("drlConfig.cfg");
    bestResult = sys.float_info.max
    test_result_file_path = os.path.join(logFolder, 'training_record.csv')
    initResultFile(test_result_file_path)

    # Generate instances
    print("Generating instances...")
    training_pool_path = os.path.join('./train_pool_randCSP./', 'temp_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    testing_pool_path = os.path.join('./test_pool_randCSP./', 'temp_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(training_pool_path):
        os.makedirs(training_pool_path)
    training_pool_path = training_pool_path + "./"
    if not os.path.exists(testing_pool_path):
        os.makedirs(testing_pool_path)
    testing_pool_path = testing_pool_path + "./"
    RBGenerator.genInstanceFiles(k, n, alpha, r, p, training_pool_path, training_pool_size, ifSat);
    RBGenerator.genInstanceFiles(k, n, alpha, r, p, testing_pool_path, testing_pool_size, ifSat);

    # Warm up
    for i in range(warmup_instance_num):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"-----Python: warm up round "+str(i)+"-----");
        ranInsID=randint(0, training_pool_size - 1);
        trainCasePath= training_pool_path + str(ranInsID) + ".txt";
        solver, varList, timeLimit=solverConstructor_RBGenerator.constructSolver_WarmUp(ranInsID, trainCasePath, masterDRL);
        # Do the search, and collect transactions to populate nStepMem
        timeUsed = solveInstance(solver, varList)
        print("Time used=" + str(timeUsed) + "\n")
    print("Warm up ended.\n");

    # Load the pretrained model
    if use_pre_Train:
        print("Use pretrained model, path is:" + pretrained_model_path)
        pywrapcp.Solver.loadDRLModel(masterDRL, os.path.join(pretrained_model_path, 'model'))

    pywrapcp.Solver.takeSnapshot(masterDRL)

    # Training
    for i in range(training_pool_size):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"-----Python: training round "+str(i)+"-----");
        trainCasePath = training_pool_path + str(i) + ".txt";
        solver, varList, timeLimit = solverConstructor_RBGenerator.constructSolver_Training(i, trainCasePath, masterDRL);

        # Do the search, and train the DQN
        timeUsed = solveInstance(solver, varList)
        print("Time used=" + str(timeUsed) + "\n")

        if (i>0 and (i + 1) % 50 == 0): # Do the test and save model
            test(masterDRL, i + 1, test_result_file_path)
            pywrapcp.Solver.saveDRLModel(masterDRL, str(os.path.join(logFolder, 'model_'+str(i+1))));
            print("Model updated.")

        if (i>0 and i % 100 == 0):
            pywrapcp.Solver.takeSnapshot(masterDRL)

    pywrapcp.Solver.deleteMasterDRL(masterDRL);

    # Detele instances
    shutil.rmtree(training_pool_path)
    shutil.rmtree(testing_pool_path)

    print('Done.')