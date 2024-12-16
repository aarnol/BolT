
# parse the arguments

import argparse
import torch
from datetime import datetime
import os

from utils import Option, metricSummer, calculateMetrics, dumpTestResults

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="abide1")
parser.add_argument("-m", "--model", type=str, default="bolT")
parser.add_argument("-a", "--analysis", type=bool, default=False)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default="noname")
parser.add_argument("--test_only", type=bool, default=False)



argv = parser.parse_args()



from Dataset.datasetDetails import datasetDetailsDict

# import model runners

from Models.SVM.run import run_svm
from Models.BolT.run import run_bolT 
# import hyper param fetchers

from Models.SVM.hyperparams import getHyper_svm
from Models.BolT.hyperparams import getHyper_bolT
import pickle

hyperParamDict = {

        "svm" : getHyper_svm,
        "bolT" : getHyper_bolT,

}

modelDict = {

        "svm" : run_svm,
        "bolT" : run_bolT,
}


getHyper = hyperParamDict[argv.model]
runModel = modelDict[argv.model]

print("\nTest model is {}".format(argv.model))


datasetName = argv.dataset
datasetDetails = datasetDetailsDict[datasetName]
hyperParams = getHyper()
print("Dataset details : {}".format(datasetDetails))

# test

if(datasetName == "abide1"):
    seeds = [0,1,2,3,4]
else:
    seeds = [0]
    
resultss = []

for i, seed in enumerate(seeds):

    # for reproducability
    torch.manual_seed(seed)

    print("Running the model with seed : {}".format(seed))
    if(argv.model == "bolT"):
        results = runModel(hyperParams, Option({**datasetDetails,"datasetSeed":seed}), device="cuda:{}".format(argv.device), analysis=argv.analysis)
    else:
        results = runModel(hyperParams, Option({**datasetDetails,"datasetSeed":seed}), device="cuda:{}".format(argv.device))

    resultss.append(results)
    


# metricss = calculateMetrics(resultss) 
# meanMetrics_seeds, stdMetrics_seeds, meanMetric_all, stdMetric_all = metricSummer(metricss, "test")

# # now dump metrics
# dumpTestResults(argv.name, hyperParams, argv.model, datasetName, metricss)

# print("\n \ n meanMetrics_all : {}".format(meanMetric_all))
# print("stdMetric_all : {}".format(stdMetric_all))
# print(resultss)
if(argv.analysis):
    analysis_path = "Analysis/Logs/{}_{}_{}_{}".format(argv.name, argv.model, datasetName, datetime.now().strftime("%Y%m%d"))
    if(not os.path.exists(analysis_path)):
        os.makedirs(analysis_path)

    fold_0_results = resultss[0][0]
    step_metrics = fold_0_results["train"]["step_metrics"]

    # Save step metrics to a file
    with open(os.path.join(analysis_path, 'step_metrics.pkl'), 'wb') as f:
        pickle.dump(step_metrics, f)

    print("Step metrics saved to {}/step_metrics.pkl".format(analysis_path))

    # Save epoch metrics to a file
    epoch_metrics = fold_0_results["train"]["epoch_metrics"]
    with open(os.path.join(analysis_path, 'epoch_metrics.pkl'), 'wb') as f:
        pickle.dump(epoch_metrics, f)

    print("Epoch metrics saved to {}/epoch_metrics.pkl".format(analysis_path))

    # Save test metrics to a file
    test_metrics = fold_0_results["test"]["metrics"]
    with open(os.path.join(analysis_path, 'test_metrics.pkl'), 'wb') as f:
        pickle.dump(test_metrics, f)

    print("Test metrics saved to {}/test_metrics.pkl".format(analysis_path))

    # Save test results to a file
    test_results = fold_0_results["test"]["results"]
    with open(os.path.join(analysis_path, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f)
    
    print("Test results saved to {}/test_results.pkl".format(analysis_path))

    # save hyper params
    hyperParamFile = open(analysis_path + "/" + "hyperParams.txt", "w")
    for key in vars(hyperParams):
        hyperParamFile.write("\n{} : {}".format(key, vars(hyperParams)[key]))
    hyperParamFile.close()


