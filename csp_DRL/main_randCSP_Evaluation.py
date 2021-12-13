import sys
sys.path.append('../or-tools-6.7.2-customized/ortools/gen')

import RBGenerator
import solverConstructor_RBGenerator
from ortools.constraint_solver import pywrapcp
import csv, datetime, os, shutil

# Pretrained model
pretrained_model_name = '3_15'
pretrained_model_path = os.path.join('./pre_trained_models', pretrained_model_name)
max_infer_depth = -1    # Set the maximum number of DNN inference depth ($\mathcal{K}$ in the paper)

# Evaluation pool setting
evaluation_pool_size = 500
use_existing_pool = True
exising_pool_name = '3_25'
existing_pool_path = os.path.join('./evaluation_pool_randCSP', exising_pool_name)

# RBGenerator config - generate instances using the below setting if use_existing_pool == False
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

# Baselines
baselines = {}
baselines = {6:'MinDomTdeg'}
# baselines = {0:'MinDom', 1:'MinDomDdeg', 3:'Default', 6:'MinDomTdeg'}
resultFieldnames = ['CaseID', 'RealTime', 'InferTime' ,'TimeOut', 'nDecisions', 'nBranches', 'nFails']

# CP solver config
solverConstructor_RBGenerator.useTimeLimit = False
solverConstructor_RBGenerator.solverNodeLimit = 50 * 10000
solverConstructor_RBGenerator.solverTimeLimit = 60 * 1000


# Initialize the result file
def initResultFile(logFolder):
    # Create result file for DQN
    with open(os.path.join(logFolder, 'result_DQN.csv'), 'w', newline='') as csvfile:
        csvWriter = csv.DictWriter(csvfile, resultFieldnames)
        csvWriter.writeheader()
    # Create result files for the baselines
    for key, value in baselines.items():
        with open(os.path.join(logFolder, 'result_'+value+'.csv'), 'w', newline='') as csvfile:
            csvWriter = csv.DictWriter(csvfile, resultFieldnames)
            csvWriter.writeheader()

# Solve an instance and write the result
def solveInstance(solver, varList, searchLimit, caseID, resultFilePath):
    solver.NextSolution_DRL()
    for var in varList:
        print(var, end=" ")
    print('')
    solver.EndSearch()
    # Record performance
    realTime = solver.getRealSolvingTime()
    timeOut = 0 if searchLimit.Crossed() == False else 1
    inferTime = solver.getInferenceTime()
    nDecisions = solver.getNumDecisions()
    nBranches = solver.Branches()
    nFails = solver.Failures()
    # Write result
    recordDict = {'CaseID':str(caseID), 'RealTime':str(realTime), 'InferTime':str(inferTime), 'TimeOut':str(timeOut),
              'nDecisions':str(nDecisions), 'nBranches':str(nBranches), 'nFails':str(nFails)}
    with open(resultFilePath, 'a', newline='') as csvfile:
        csvWriter = csv.DictWriter(csvfile, resultFieldnames)
        csvWriter.writerow(recordDict)
    # Print the result to console
    for key, value in recordDict.items():
        print(key+'='+value, end=' ')
    print('\n')


# Modify the max_infer_depth parameter of the config file
def modifyConfig(train_config, logFolder):
    # Read and modify old file
    new_config_file_path = os.path.join(logFolder, "drlConfig.cfg")
    with open(train_config) as config_file:
        lines = config_file.readlines()
        lastline = lines[len(lines)-1]
        last_config = lastline.split()
        lines[len(lines)-1] = 'max_infer_depth '+str(max_infer_depth)
        # Write new file
        with open(new_config_file_path, 'w') as config_file_new:
            config_file_new.writelines(lines)
        print('Config file modified.')


def createOverallResult(logFolder):
    overallResultFields = resultFieldnames.copy()
    overallResultFields[0] = 'Name'
    with open(os.path.join(logFolder, 'result_Overall.csv'), 'w', newline='') as csvfile:
        csvWriter = csv.DictWriter(csvfile, overallResultFields)
        csvWriter.writeheader()
        # Get the overall result of DQN
        sumDict_DQN = {}
        for item in resultFieldnames:
            if item != 'CaseID':
                sumDict_DQN[item] = 0.0
        with open(os.path.join(logFolder, 'result_DQN.csv'), newline='') as result_DQN:
            csvReader_DQN = csv.DictReader(result_DQN)
            for row in csvReader_DQN:
                for key, value in sumDict_DQN.items():
                    sumDict_DQN[key] = value + float(row[key])
            resultDict_DQN = {'Name': 'DQN'}
            for key, value in sumDict_DQN.items():
                resultDict_DQN[key] = sumDict_DQN[key]/evaluation_pool_size
            csvWriter.writerow(resultDict_DQN)
        # Get the overall result for each baseline
        for baselineID, baselineName in baselines.items():
            sumDict_BS = {}
            for item in resultFieldnames:
                if item != 'CaseID':
                    sumDict_BS[item] = 0.0
            with open(os.path.join(logFolder, 'result_'+baselineName+'.csv'), newline='') as result_BS:
                csvReader_BS = csv.DictReader(result_BS)
                for row in csvReader_BS:
                    for key, value in sumDict_BS.items():
                        sumDict_BS[key] = value + float(row[key])
                resultDict_BS = {'Name': baselineName}
                for key, value in sumDict_BS.items():
                    resultDict_BS[key] = sumDict_BS[key] / evaluation_pool_size
                csvWriter.writerow(resultDict_BS)


if __name__=='__main__':

    # Create and initialize the log folder
    log_folder = os.path.join("./log", 'evaluation_log_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    initResultFile(log_folder)

    # Generate instances
    if (use_existing_pool == False):
        print("Generating new instances...")
        evaluation_pool_path = os.path.join('./evaluation_pool_randCSP', 'temp_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(evaluation_pool_path):
            os.makedirs(evaluation_pool_path)
        evaluation_pool_path = evaluation_pool_path + './'
        RBGenerator.genInstanceFiles(k, n, alpha, r, p, evaluation_pool_path, evaluation_pool_size, ifSat);
        # Save RBConfig
        with open(os.path.join(log_folder, 'RBConfig.txt'), 'w') as RBConfigFile:
            RBConfigFile.write("k\tn\talpha\tr\tp\n")
            RBConfigFile.write(str(k) + "\t" + str(n) + "\t" + str(alpha) + "\t" + str(r) + "\t" + str(p) + "\n")
    else:
        print("Using existing pool " + existing_pool_path)
        evaluation_pool_path = existing_pool_path + './'
        shutil.copyfile(os.path.join(evaluation_pool_path, 'RBConfig.txt'), os.path.join(log_folder, 'RBConfig.txt')) # Copy the RBConfig

    # Initialization
    modifyConfig(os.path.join(pretrained_model_path, "drlConfig.cfg"), log_folder)
    masterDRL = pywrapcp.Solver.createMasterDRL(os.path.join(log_folder, "drlConfig.cfg"));
    pywrapcp.Solver.loadDRLModel(masterDRL, os.path.join(pretrained_model_path, 'model'))
    print('Pretrained model loaded.')

    # Evaluation
    for i in range(evaluation_pool_size):
        print("-----Python: evaluation round "+str(i)+"-----");
        instance_path = evaluation_pool_path + str(i) + ".txt";

        # Solve using DQN
        solver, varList, timeLimit = solverConstructor_RBGenerator.constructSolver_Testing(i, instance_path, masterDRL);
        solveInstance(solver, varList, timeLimit, i, os.path.join(log_folder, 'result_DQN.csv'))

        # Solve using baselines
        for key, value in baselines.items():
            print("Baseline: "+value)
            solver, varList, timeLimit = solverConstructor_RBGenerator.constructSolver_Baseline(i, instance_path, masterDRL, key);
            solveInstance(solver, varList, timeLimit, i, os.path.join(log_folder, 'result_' + value + '.csv'))

    pywrapcp.Solver.deleteMasterDRL(masterDRL);

    # Create the average record
    createOverallResult(log_folder)

    # Clear the evaluation pool
    if (use_existing_pool == False):
       shutil.rmtree(evaluation_pool_path)

    print('Done.')