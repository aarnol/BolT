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

from Models.Triplet.model import Model
from Dataset.dataset import getTripletDataset
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
    losses = []
    neg_distances = []
    pos_distances = []
    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
        xTest = data

        test_loss, anchor_embeddings, pos_embeddings, negative_embeddings = model.step(xTest, train=False)
       
        # Check for NaNs in embeddings
        for name, tensor in [('anchor', anchor_embeddings), ('positive', pos_embeddings), ('negative', negative_embeddings)]:
            if torch.isnan(tensor).any():
                print(f"NaN detected in {name} embeddings at batch {i}")
            if torch.norm(tensor, dim=1).eq(0).any():
                print(f"Zero vector detected in {name} embeddings at batch {i}")

        # Clamp cosine similarity results to avoid NaNs from bad inputs
        anchor_positive_distance = torch.mean(1 - torch.nn.functional.cosine_similarity(anchor_embeddings, pos_embeddings))
        anchor_negative_distance = torch.mean(1 - torch.nn.functional.cosine_similarity(anchor_embeddings, negative_embeddings))

        # Check for NaNs in distances
        if torch.isnan(anchor_positive_distance):
            print(f"NaN in anchor-positive distance at batch {i}")
        if torch.isnan(anchor_negative_distance):
            print(f"NaN in anchor-negative distance at batch {i}")
        if torch.isnan(test_loss):
            print(f"NaN in loss at batch {i}")

        # Append only if valid
        if not (torch.isnan(anchor_positive_distance) or torch.isnan(anchor_negative_distance) or torch.isnan(test_loss)):
            pos_distances.append(anchor_positive_distance.item())
            neg_distances.append(anchor_negative_distance.item())
            losses.append(test_loss.item())

        torch.cuda.empty_cache()


    print("Fold: {} Epoch: {} Loss: {}".format(fold, epoch, np.mean(losses)))
    print("Fold: {} Epoch: {} Positive Distance: {}".format(fold, epoch, np.mean(pos_distances)))
    print("Fold: {} Epoch: {} Negative Distance: {}".format(fold, epoch, np.mean(neg_distances)))
    return np.mean(neg_distances), np.mean(pos_distances), np.mean(losses)
     
       
        
   






def run_triplet(hyperParams, datasetDetails, device="cuda:3", analysis=False, name = "noname"):


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
    