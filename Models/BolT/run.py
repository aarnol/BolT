from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys

from datetime import datetime

if(not "utils" in os.getcwd()):
    sys.path.append("../../../")


from utils import Option
from utils import Option, calculateMetric

from Models.BolT.model import Model
from Dataset.dataset import getDataset
from torch.nn.utils.rnn import pad_sequence


def train(model, dataset, fold, nOfEpochs):

    dataLoader = dataset.getFold(fold, train=True)
    test_metrics = []
    test_results = []
    step_metrics = []
    losses = []
    for epoch in range(nOfEpochs):

        preds = []
        probs = []
        groundTruths = []
        
        
        

        for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
            
            xTrain = data[0] # (batchSize, N, dynamicLength)
            yTrain = data[1] # (batchSize, )

            # NOTE: xTrain and yTrain are still on "cpu" at this point

            train_loss, train_preds, train_probs, yTrain = model.step(xTrain, yTrain, train=True)

            torch.cuda.empty_cache()

            preds.append(train_preds)
            probs.append(train_probs)
            groundTruths.append(yTrain)
            losses.append(train_loss.item())

            # Log metrics every step
            step_metric = calculateMetric({"predictions":train_preds.numpy(), "probs":train_probs.numpy(), "labels":yTrain.numpy()})
            step_metric['loss'] = train_loss.item()
            step_metrics.append(step_metric)

        preds = torch.cat(preds, dim=0).numpy()
        probs = torch.cat(probs, dim=0).numpy()
        groundTruths = torch.cat(groundTruths, dim=0).numpy()
        

        metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
        print(f"Epoch {epoch} train metrics: {metrics}")

        test_preds, test_probs, test_groundTruths, test_loss, epoch_test_metrics= test(model, dataset, fold)
        epoch_test_results = {}
        epoch_test_results['loss'] = test_loss
        epoch_test_results['predictions'] = test_preds
        epoch_test_results['probs'] = test_probs
        epoch_test_results['labels'] = test_groundTruths
        print(f"Epoch {epoch} test metrics: {epoch_test_metrics}")
        test_metrics.append(epoch_test_metrics)
        test_results.append(epoch_test_results)
    print(len(test_results))
    print(len(test_metrics))
    return preds, probs, groundTruths, losses, metrics, step_metrics, test_metrics, test_results





def test(model, dataset, fold):

    dataLoader = dataset.getFold(fold, train=False)

    preds = []
    probs = []
    groundTruths = []
    losses = []        

    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Testing fold:{fold}')):

        xTest = data["timeseries"]
        yTest = data["label"]

        # NOTE: xTrain and yTrain are still on "cpu" at this point

        test_loss, test_preds, test_probs, yTest = model.step(xTest, yTest, train=False)
        
        torch.cuda.empty_cache()

        preds.append(test_preds)
        probs.append(test_probs)
        groundTruths.append(yTest)
        losses.append(test_loss)

    preds = torch.cat(preds, dim=0).numpy()
    probs = torch.cat(probs, dim=0).numpy()
    groundTruths = torch.cat(groundTruths, dim=0).numpy()
    loss = torch.tensor(losses).numpy().mean()          

    metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
    print("\n \n Test metrics : {}".format(metrics))                
    
    return preds, probs, groundTruths, loss, metrics
    


def run_bolT(hyperParams, datasetDetails, device="cuda:3", analysis=False):


    # extract datasetDetails

    foldCount = datasetDetails.foldCount
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs


    dataset = getDataset(datasetDetails)


    details = Option({
        "device" : device,
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
        "nOfClasses" : datasetDetails.nOfClasses,
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs
    })


    results = []

    for fold in range(foldCount):

        model = Model(hyperParams, details)


        train_preds, train_probs, train_groundTruths, train_loss, epoch_metrics, step_metrics, test_metrics, test_results = train(model, dataset, fold, nOfEpochs)   
        
        result = {
            "fold" : fold,

            "train" : {
                "labels" : train_groundTruths,
                "predictions" : train_preds,
                "probs" : train_probs,
                "loss" : train_loss,
                "epoch_metrics" : epoch_metrics,
                "step_metrics" : step_metrics
            },

            "test" : {
                "results" : test_results,
                "metrics" : test_metrics
            }

        }

        results.append(result)


        if(analysis):
            targetSaveDir = "./Analysis/TargetSavedModels/{}/seed_{}/".format(datasetDetails.datasetName, datasetSeed)
            os.makedirs(targetSaveDir, exist_ok=True)
            torch.save(model, targetSaveDir + "/model_{}.save".format(fold))
        break


    return results
