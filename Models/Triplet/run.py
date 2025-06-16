from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys

from datetime import datetime
import torch.nn.functional as F
if(not "utils" in os.getcwd()):
    sys.path.append("../../../")


from utils import Option
from utils import Option, calculateMetric

from Models.Triplet.model import Model
from Dataset.dataset import getTripletDataset, getBalancedContrastiveDataset
from torch.nn.utils.rnn import pad_sequence


def train(model, dataset, fold, nOfEpochs):
   
    dataLoader = dataset.getFold(fold, train=True)
    losses = []

    for epoch in range(nOfEpochs):
        # torch.cuda.empty_cache()
        preds = []
        probs = []
        groundTruths = []
        
        
        

        for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
            
            xTrain = data
            sample1 = xTrain[0].to(model.device)  # anchor
            sample2 = xTrain[1].to(model.device)  # positive
            similarity = xTrain[2].to(model.device)  
            # NOTE: xTrain and yTrain are still on "cpu" at this point

            train_loss, _, _, _ = model.step(xTrain, train=True)

            torch.cuda.empty_cache()

         
            losses.append(train_loss.item())
            if(i % 100 == 0):
                print("Epoch: {} Fold: {} Iteration: {} Loss: {}".format(epoch, fold, i, train_loss.item()))
        if epoch % 2 == 0:
            test(model, dataset, fold, epoch)    
  
    return losses

def test(model, dataset, fold, epoch):
    #calculate the distance between the embeddings of the anchor, positive and negative samples
    # and return the loss
   
    dataLoader = dataset.getFold(fold, train=False)
    
    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
        xTest = data

        loss_value, embeddings_cpu, labels, modalities = model.step(xTest, train=False)

        raise Exception("Not Implemented Yet")
    return None
        
        
        

       
        
   






def run_triplet(hyperParams, datasetDetails, device="cuda:3", analysis=False, name = "noname"):


    # extract datasetDetails

   
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs
    foldCount = datasetDetails.foldCount


    dataset = getBalancedContrastiveDataset(datasetDetails)


    details = Option({
        "device" : device,
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs
    })


    results = []

    for fold in range(foldCount):
        model = Model(hyperParams, details)
        print('training from scratch')

        train_metrics= train(model, dataset, fold, nOfEpochs)
        result = {
            "fold" : fold,

            "train_metrics" : train_metrics,
            

        }

   

        if(analysis):
            targetSaveDir = "./Analysis/TargetSavedModels/{}/{}/seed_{}/".format(datasetDetails.datasetName, name, datasetSeed)
            os.makedirs(targetSaveDir, exist_ok=True)
            torch.save(model, targetSaveDir + "/model_{}.save".format(fold))
        
        
        
    

    return results


def test_triplet(hyperParams, train, test, fold, datasetDetails, device="cuda:3", analysis=False, name = "noname"):


    # extract datasetDetails

   
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs
    foldCount = datasetDetails.foldCount


    dataset = getTripletDataset(datasetDetails)


    details = Option({
        "device" : device,
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs
    })
    