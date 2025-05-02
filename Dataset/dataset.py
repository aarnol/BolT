
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import GroupKFold
from random import shuffle, randrange
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
#from .DataLoaders.hcpRestLoader import hcpRestLoader
#from .DataLoaders.hcpTaskLoader import hcpTaskLoader
#from .DataLoaders.abide1Loader importabide1Loader
from .DataLoaders.hcpWorkingMemLoader import hcpWorkingMemLoader
from .DataLoaders.hcpfNIRSLoader import hcpfNIRSLoader
from .DataLoaders.hcpMotorLoader import hcpMotorLoader

loaderMapper = {
    #"hcpRest" : hcpRestLoader,
    #"hcpTask" : hcpTaskLoader,
    #"abide1" : abide1Loader,
    "hcpWM": hcpWorkingMemLoader,
    "hcpfNIRS" : hcpfNIRSLoader,
    "hcpMotor" : hcpMotorLoader
}
def custom_collate_fn(batch):
    """
    Custom collate function to pad or truncate sequences to the same length.
    
    Args:
        batch (list of dict): A batch of data where each item is a dictionary 
                              with keys 'timeseries' and 'label'.
    
    Returns:
        inputs_padded (Tensor): Padded input sequences, shape (batch_size, max_seq_length, feature_dim).
        labels_stacked (Tensor): Stacked labels, shape (batch_size, ...).
    """
    # Extract 'timeseries' and 'label' tensors
    inputs = [torch.tensor(item['timeseries']) for item in batch]
    labels = [torch.tensor(item['label']) for item in batch]

    # Find the maximum time dimension in the batch
    max_time = max(seq.size(1) for seq in inputs)

    # Pad or truncate all sequences to the same time dimension
    aligned_inputs = []
    for seq in inputs:
        if seq.size(1) < max_time:  # Pad if shorter
            padding = torch.zeros(seq.size(0), max_time - seq.size(1))
            aligned_inputs.append(torch.cat([seq, padding], dim=1))
        elif seq.size(1) > max_time:  # Truncate if longer
            aligned_inputs.append(seq[:, :max_time])
        else:  # Already aligned
            aligned_inputs.append(seq)
    
    # Stack aligned inputs
    inputs_padded = torch.stack(aligned_inputs)

    # Stack labels (assuming they are fixed-sized tensors)
    labels_stacked = torch.stack(labels)

    return inputs_padded, labels_stacked
def getDataset(options):
    return SupervisedDataset(options)

def guassianNoise(data, mean=0, std=0.1):
    """
    Add Gaussian noise to the data.
    
    Args:
        data (numpy.ndarray): Input data.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    
    Returns:
        numpy.ndarray: Data with added Gaussian noise.
    """
    noise = np.random.normal(mean, std, data.shape)
    return data + noise


class SupervisedDataset(Dataset):
    
    def __init__(self, datasetDetails):

        self.batchSize = datasetDetails.batchSize
        self.dynamicLength = datasetDetails.dynamicLength
        self.foldCount = datasetDetails.foldCount

        self.seed = datasetDetails.datasetSeed

        loader = loaderMapper[datasetDetails.datasetName]

        self.kFold = GroupKFold(datasetDetails.foldCount) if datasetDetails.foldCount is not None else None
        self.k = None
        
        self.fnirs = datasetDetails.fNIRS
        
        if(self.fnirs):
            self.signal = datasetDetails.signal
            #self.subject = datasetDetails.subject

        if self.fnirs:
            self.data, self.labels, self.subjectIds = loader(datasetDetails.atlas, datasetDetails.targetTask, datasetDetails.signal, datasetDetails.subject)
        else:
            self.data, self.labels, self.subjectIds = loader(datasetDetails.atlas, datasetDetails.targetTask)
        

        # Filter out samples where the last axis is smaller than dynamicLength
        # valid_data_indices = [idx for idx, subject in enumerate(self.data) if subject.shape[-1] >= self.dynamicLength]
        # self.data = [self.data[idx] for idx in valid_data_indices]
        # self.labels = [self.labels[idx] for idx in valid_data_indices]
        # self.subjectIds = [self.subjectIds[idx] for idx in valid_data_indices]
        
        self.groups = list([int(str(subject)) for subject in self.subjectIds])
        
        

        self.targetData = None
        self.targetLabel = None
        self.targetSubjIds = None

        self.randomRanges = None

        self.trainIdx = None
        self.testIdx = None

    def __len__(self):
        return len(self.data) if isinstance(self.targetData, type(None)) else len(self.targetData)

    def get_nOfTrains_perFold(self):
        if(self.foldCount != None):
            return int(np.ceil(len(self.data) * (self.foldCount - 1) / self.foldCount))           
        else:
            return len(self.data)        

    def setFold(self, fold, train=True):

        self.k = fold
        self.train = train

        if(self.foldCount == None):  # if this is the case, TEST must be True (different behavior) than original BolT
            testIdx = list(range(len(self.data)))
            self.targetData = [self.data[idx] for idx in testIdx]
            self.targetLabels =  [self.labels[idx] for idx in testIdx]
            self.targetSubjIds =  [self.subjectIds[idx] for idx in testIdx]
            return





        else:
            splits = list(self.kFold.split(self.data, self.labels, self.groups))
            trainIdx, testIdx = splits[fold]

            # Ensure groups are not shared between train and test
            
            train_groups = set([self.groups[idx] for idx in trainIdx])
            test_groups = set([self.groups[idx] for idx in testIdx])
            intersect = train_groups.intersection(test_groups)
            if intersect:
                raise ValueError(f"Group(s) {intersect} found in both train and test splits for fold {fold}!")
            else:
                print(f"Fold {fold} is valid.")
      

        self.trainIdx = trainIdx
        self.testIdx = testIdx

        random.Random(self.seed).shuffle(trainIdx)

        self.targetData = [self.data[idx] for idx in trainIdx] if train else [self.data[idx] for idx in testIdx]
        self.targetLabels = [self.labels[idx] for idx in trainIdx] if train else [self.labels[idx] for idx in testIdx]
        self.targetSubjIds = [self.subjectIds[idx] for idx in trainIdx] if train else [self.subjectIds[idx] for idx in testIdx]
        train_groups = set([self.groups[idx] for idx in trainIdx])
        test_groups = set([self.groups[idx] for idx in testIdx])
        intersect = train_groups.intersection(test_groups)
        if(train):
            train_subjects = set(self.targetSubjIds)
            val_subjects = set(self.subjectIds[idx] for idx in self.testIdx)
            if train_subjects & val_subjects:
                print(train_subjects & val_subjects)
                raise ValueError("Train and test subjects are leaking!")
            
            else:
                print("Train and test subjects are properly separated.")
        if(not train):
            train_subjects = set(self.subjectIds[idx] for idx in self.trainIdx)
            val_subjects = set(self.subjectIds[idx] for idx in self.testIdx)
            if train_subjects & val_subjects:
                print(train_subjects & val_subjects)
                raise ValueError("Train and test subjects are leaking!")
            
            else:
                print("Train and test subjects are properly separated.")


        # if train:
        #     print("RANDOM LABELS")
        #     #random labels
        #     self.targetLabels = [random.choice([0, 1]) for _ in range(len(self.targetLabels))]  # random labels for training
            
            
        if intersect:
            print(f"Groups leaking between train and test: {intersect}")
            exit()
        else:
            print("No leakage between train and test groups.")

        if(train and not isinstance(self.dynamicLength, type(None))):
            np.random.seed(self.seed+1)
            
                
            self.randomRanges = [[np.random.randint(0, self.data[idx].shape[-1] - self.dynamicLength) for k in range(9999)] for idx in trainIdx]

    def getFold(self, fold, train=True):

        self.setFold(fold, train)
        

        if(train):
            return DataLoader(self, batch_size=self.batchSize, shuffle=False, collate_fn = custom_collate_fn)
        else:
           
            return DataLoader(self, batch_size=1, shuffle=False)            

    def __getitem__(self, idx):
        subject = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]

        # normalize timeseries
        timeseries = subject  # (numberOfRois, time)
        
        # #z score
        #timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1, keepdims=True)

        #normalize to between 0 and 1
        #timeseries = (timeseries - np.min(timeseries, axis=1, keepdims=True)) / (np.max(timeseries, axis=1, keepdims=True) - np.min(timeseries, axis=1, keepdims=True))



        timeseries = np.nan_to_num(timeseries, 0)
        #check if timeseries is zscored properly
        
        
        # dynamic sampling if train
        if(self.train and not isinstance(self.dynamicLength, type(None))):

            samplingInit = self.randomRanges[idx].pop()

            timeseries = timeseries[:, samplingInit : samplingInit + self.dynamicLength]
            
            # # add noise
            # timeseries = guassianNoise(timeseries, mean=0, std=0.1)
        
        
        return {"timeseries" : timeseries.astype(np.float32), "label" : label, "subjId" : subjId}






