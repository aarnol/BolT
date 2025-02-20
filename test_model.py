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
parser.add_argument("-m", "--model_path", type=str)
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
model_path = os.path.join(os.getcwd(), "Analysis", "TargetSavedModels", "hcpWM", argv.model_path,"seed_0", "model_0.save")
device = argv.device
model = torch.load(model_path)
dataset = getDataset(Option({**datasetDetails,"datasetSeed":seed}))
results = test(model, dataset, None)

#generate confusion matrix
preds = results[0]
labels = results[2]

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = confusion_matrix(labels, preds)
classes = ["0back", "2back"]    
# Create the heatmap
# plt.figure(figsize=(5, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)

# Labels and title
# plt.xlabel('Predicted Label')
# plt.ylabel('Actual Label')
# plt.title('15mm Sphere fNIRS Confusion Matrix')

# plt.show()