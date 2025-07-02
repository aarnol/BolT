from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys

from datetime import datetime
import torch.nn.functional as F
from time import sleep
if(not "utils" in os.getcwd()):
    sys.path.append("../../../")


from utils import Option
from utils import Option, calculateMetric

from Models.Triplet.model import Model, Classifier
from Dataset.dataset import getTripletDataset, getBalancedContrastiveDataset, getDataset
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

            train_loss, e1, e2, similarities = model.step(xTrain, train=True)

            torch.cuda.empty_cache()

         
            losses.append(train_loss.item())
            if i % 100 == 0:
                with open("losses.txt", "a") as f:
                    f.write("Epoch: {} Fold: {} Loss: {}\n".format(epoch, fold, np.mean(losses)))
        if epoch % 2 == 0:
            test(model, dataset, fold, epoch)    
    print("Epoch: {} Fold: {} Loss: {}".format(epoch, fold, np.mean(losses)))
    
    return losses

def test(model, dataset, fold, epoch):
    
   
    dataLoader = dataset.getFold(fold, train=False)
    losses = []
    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
        xTest = data

        test_loss, e1, e2, similarities = model.step(xTest, train=False)
        torch.cuda.empty_cache()
        losses.append(test_loss.item())
    test_loss = torch.tensor(losses).mean()
    print("Epoch: {} Fold: {} Test Loss: {}".format(epoch, fold, test_loss.item()))
    return test_loss.item()


       
        
   






def run_triplet(hyperParams, datasetDetails, device="cuda:3", analysis=False, name = "noname"):


    # extract datasetDetails

   
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs
    foldCount = datasetDetails.foldCount


    dataset = getBalancedContrastiveDataset(datasetDetails)
    print("Dataset loaded with seed: {}".format(datasetSeed))

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

def run_classifier(hyperParams, datasetDetails, device="cuda:3", analysis=False, name = "noname", train = True):



    # extract datasetDetails
    
   
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs
    foldCount = datasetDetails.foldCount
    
    dataset = getDataset(datasetDetails)
    print("Dataset loaded with seed: {}".format(datasetSeed))

    details = Option({
        "device" : device,
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs,
        "nOfClasses" : datasetDetails.nOfClasses,
    })

    results = []

    for fold in range(foldCount):
        e_model_path = './Analysis/TargetSavedModels/hcpfNIRS/triplettest/seed_{}/model_{}.save'.format(datasetSeed, fold)
        

        embedding_model = torch.load(e_model_path)
        
        model = Classifier(hyperParams, details)

        print('training from scratch')

        if train is not None:
            train_metrics = train_classifier(model, embedding_model, dataset, fold, nOfEpochs)
            result = {
                "fold" : fold,
                "train_metrics" : train_metrics,
            }
        

        results.append(result)

        if(analysis):
            targetSaveDir = "./Analysis/TargetSavedModels/{}/{}/seed_{}/".format(datasetDetails.datasetName, name, datasetSeed)
            os.makedirs(targetSaveDir, exist_ok=True)
            torch.save(embedding_model, targetSaveDir + "/model_{}.save".format(fold))

def train_classifier(model, embedding_model, dataset, fold, nOfEpochs):
    dataLoader = dataset.getFoldFromFile(fold, train=True)

    losses = []

    for epoch in range(nOfEpochs):
        # torch.cuda.empty_cache()
        preds = []
        probs = []
        groundTruths = []
        
        

        for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
            if i == 0:
                print(len(data))
            xTrain, yTrain = data
            # NOTE: xTrain and yTrain are still on "cpu" at this point
            
            xTrain = embedding_model.getEmbeddings(xTrain)
           
            loss_value, predictions, labels = model.step((xTrain,yTrain), train=True)

            torch.cuda.empty_cache()
            
            preds.append(predictions)
            groundTruths.append(labels)

            losses.append(loss_value.item())
            if i % 100 == 0:
                with open("losses.txt", "a") as f:
                    f.write("Epoch: {} Fold: {} Loss: {}\n".format(epoch, fold, np.mean(losses)))
        
        #compute accuracy
        preds = torch.cat(preds, dim=0).numpy()
        groundTruths = torch.cat(groundTruths, dim=0).numpy()
        loss = torch.tensor(losses).numpy().mean()
        results = {"predictions": preds, "labels": groundTruths, "loss": loss}
        metrics = calculateMetric(results)
       
        preds = np.argmax(preds, axis=1)
        
        #calculate accuracy
        accuracy = np.mean(preds == groundTruths)
        print(f'Epoch: {epoch} Fold: {fold} Accuracy: {accuracy:.4f}', flush=True)
        print("Epoch: {} Fold: {} Metrics: {}".format(epoch, fold, metrics))
        if epoch % 2 == 0:
            test_classifier(model,embedding_model, dataset, fold, epoch)
       
    print("Epoch: {} Fold: {} Loss: {}".format(epoch, fold, np.mean(losses)), flush = True)
    
    return losses

def test_classifier(model,e_model,  dataset, fold, epoch):
    
   
    dataLoader = dataset.getFoldFromFile(fold, train=False)
    losses = []
    p = []
    l = []
    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
        xTest = data['timeseries']
        yTest = data['label']
        xTest = e_model.getEmbeddings(xTest)
        
        test_loss, predictions, labels = model.step((xTest,yTest), train=False)
        torch.cuda.empty_cache()
        losses.append(test_loss.item())
        p.append(predictions)
        l.append(labels)
    test_loss = torch.tensor(losses).mean()
    p = torch.cat(p, dim=0).numpy()
    l = torch.cat(l, dim=0).numpy()
    accuracy = np.mean(np.argmax(p, axis=1) == l)
    print("Epoch: {} Fold: {} Test Loss: {} Accuracy: {}".format(epoch, fold, test_loss.item(), accuracy), flush= True)
    print("Epoch: {} Fold: {} Test Loss: {}".format(epoch, fold, test_loss.item()))
    return test_loss.item()