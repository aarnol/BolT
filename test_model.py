# Description: Test the model on the test set and save the results.
import argparse
import torch
from datetime import datetime
import os


from utils import Option, metricSummer, calculateMetrics, dumpTestResults

from Dataset.datasetDetails import datasetDetailsDict
from Dataset.dataset import getDataset
parser = argparse.ArgumentParser()


parser.add_argument("-s", "--subject", type=int, default=1)
parser.add_argument("-a", "--atlas", type=str, default="sphere15mm")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default="noname")
parser.add_argument("-dl", "--dynamic_length", type=int, default=30)

argv = parser.parse_args()

from Dataset.datasetDetails import datasetDetailsDict

# import model runners
from Models.BolT.run import test

# import hyper param fetchers
from Models.BolT.hyperparams import getHyper_bolT

import pickle
import pandas as pd

window = getHyper_bolT().windowSize
saved = pd.DataFrame(columns=["Atlas", "Window", "Dynamic Length","Signal", "Invert", "Accuracy","Precision", "Recall", "ROC", "Subject"])
for signal in ["HbT", "HbR", "HbO"]:
    for invert in [True, False]:
        print(f"Running {signal} with invert {invert}")
        dataset = f"hcpfNIRS_{argv.subject}_{signal}_{argv.dynamic_length}"
        datasetDetails = datasetDetailsDict[dataset]
        seed = 0
        model_name = f"{argv.atlas}sub{argv.subject}_{window}w_{argv.dynamic_length}dl"
        model_path = os.path.join(os.getcwd(), "Analysis", "TargetSavedModels", "hcpWM", model_name,"seed_0", "model_0.save")
        device = argv.device
        model = torch.load(model_path, weights_only=False)
        dataset = getDataset(Option({**datasetDetails,"datasetSeed":seed}))
        results = test(model, dataset, None, invert=invert)

        #generate confusion matrix
        preds = results[0]
        labels = results[2]
        metrics = results[4]
        accuracy = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        roc = metrics["roc"]

        saved.loc[len(saved)] = [argv.atlas, window, argv.dynamic_length, signal, invert, accuracy, precision, recall, roc, argv.subject]

saved.to_csv("results.csv", index=False)



















        # from sklearn.metrics import confusion_matrix
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import seaborn as sns

        # conf_matrix = confusion_matrix(labels, preds)
        # classes = ["0back", "2back"]    
# Create the heatmap
# plt.figure(figsize=(5, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)

# Labels and title
# plt.xlabel('Predicted Label')
# plt.ylabel('Actual Label')
# plt.title('15mm Sphere fNIRS Confusion Matrix')

# plt.show()