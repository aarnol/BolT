# Description: Test the model on the test set and save the results.
import argparse
import torch
from datetime import datetime
import os

from utils import Option, metricSummer, calculateMetrics, dumpTestResults

from Dataset.datasetDetails import datasetDetailsDict
from Dataset.dataset import getDataset
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="hcpWM_fNIRS")
parser.add_argument("-m", "--model_path", type=str, default=os.path.join(os.getcwd(), "Analysis", "TargetSavedModels", "hcpWM", "seed_0", "model_0.save"))
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default="noname")

argv = parser.parse_args()

from Dataset.datasetDetails import datasetDetailsDict

# import model runners
from Models.BolT.run import test

# import hyper param fetchers
from Models.BolT.hyperparams import getHyper_bolT

import pickle

hyperParams = getHyper_bolT()
datasetDetails = datasetDetailsDict[argv.dataset]

seed = 0
model_path = argv.model_path
device = argv.device
model = torch.load(model_path)
dataset = getDataset(datasetDetails)
results = test(model, datasetDetails, 0)
print(results)