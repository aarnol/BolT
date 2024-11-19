
# parse the arguments

import argparse
import torch
from datetime import datetime


from utils import Option, metricSummer, calculateMetrics, dumpTestResults

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="abide1")
parser.add_argument("-m", "--model", type=str, default="bolT")
parser.add_argument("-a", "--analysis", type=bool, default=False)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default="noname")



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
fold_0_results = resultss[0][0]
step_metrics = fold_0_results["train"]["step_metrics"]

# Save step metrics to a file
with open('step_metrics.pkl', 'wb') as f:
    pickle.dump(step_metrics, f)

print("Step metrics saved to step_metrics.pkl")

#save epoch metrics to a file
epoch_metrics = fold_0_results["train"]["epoch_metrics"]
with open('epoch_metrics.pkl', 'wb') as f:
    pickle.dump(epoch_metrics, f)

print("Epoch metrics saved to epoch_metrics.pkl")

#save test metrics to a file
test_metrics = fold_0_results["test"]["metrics"]
with open('test_metrics.pkl', 'wb') as f:
    pickle.dump(test_metrics, f)

print("Test metrics saved to test_metrics.pkl")

#save test results to a file
test_results = fold_0_results["test"]["results"]
with open('test_results.pkl', 'wb') as f:
    pickle.dump(test_results, f)


