
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
import os
from copy import deepcopy
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

        if self.foldCount is None:
            testIdx = list(range(len(self.data)))
            self.trainIdx, self.testIdx = [], testIdx  # Make sure indices are defined
            self.targetData = [self.data[idx] for idx in testIdx]
            self.targetLabels = [self.labels[idx] for idx in testIdx]
            self.targetSubjIds = [self.subjectIds[idx] for idx in testIdx]
            return

        else:
            splits = list(self.kFold.split(self.data, self.labels, self.groups))
            trainIdx, testIdx = splits[fold]

            train_groups = set(self.groups[idx] for idx in trainIdx)
            test_groups = set(self.groups[idx] for idx in testIdx)
            intersect = train_groups.intersection(test_groups)

            if intersect:
                raise ValueError(f"Group(s) {intersect} found in both train and test splits for fold {fold}!")
            else:
                print(f"Fold {fold} is valid.")

            self.trainIdx = trainIdx.copy()
            self.testIdx = testIdx

            rng = random.Random(self.seed)
            rng.shuffle(self.trainIdx)

            idx_to_use = self.trainIdx if train else self.testIdx

            self.targetData = [self.data[idx] for idx in idx_to_use]
            self.targetLabels = [self.labels[idx] for idx in idx_to_use]
            self.targetSubjIds = [self.subjectIds[idx] for idx in idx_to_use]

            train_subjects = set(self.subjectIds[idx] for idx in self.trainIdx)
            test_subjects = set(self.subjectIds[idx] for idx in self.testIdx)

            if train:
                if train_subjects & test_subjects:
                    print(train_subjects & test_subjects)
                    raise ValueError("Train and test subjects are leaking!")
                else:
                    print("Train and test subjects are properly separated.")
            else:
                if train_subjects & test_subjects:
                    print(train_subjects & test_subjects)
                    raise ValueError("Train and test subjects are leaking!")
                else:
                    print("Train and test subjects are properly separated.")

            # if train:
            #     print("RANDOM LABELS")
            #     self.targetLabels = [random.choice([0, 1]) for _ in range(len(self.targetLabels))]

            if intersect:
                print(f"Groups leaking between train and test: {intersect}")
                exit()
            else:
                print("No leakage between train and test groups.")

            if train and self.dynamicLength is not None:
                np.random.seed(self.seed + 1)
                self.randomRanges = [
                    [np.random.randint(0, self.data[idx].shape[-1] - self.dynamicLength) for _ in range(9999)]
                    for idx in self.trainIdx
                ]

    def getFold(self, fold, train=True):
        # Clone a clean copy of the dataset object to avoid state leaks
        dataset_copy = deepcopy(self)
        dataset_copy.setFold(fold, train=train)

        if train:
            return DataLoader(dataset_copy, batch_size=dataset_copy.batchSize, shuffle=False, collate_fn=custom_collate_fn)
        else:
            return DataLoader(dataset_copy, batch_size=1, shuffle=False)
           

    def __getitem__(self, idx):
        subject = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]

        # normalize timeseries
        timeseries = subject  # (numberOfRois, time)
        
        # #z score
        timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1, keepdims=True)

        #normalize to between 0 and 1
        #timeseries = (timeseries - np.min(timeseries, axis=1, keepdims=True)) / (np.max(timeseries, axis=1, keepdims=True) - np.min(timeseries, axis=1, keepdims=True))



        timeseries = np.nan_to_num(timeseries, 0)
        
        
        
        # dynamic sampling if train
        if(self.train and not isinstance(self.dynamicLength, type(None))):

            samplingInit = self.randomRanges[idx].pop()

            timeseries = timeseries[:, samplingInit : samplingInit + self.dynamicLength]
            
            # add noise
            timeseries = guassianNoise(timeseries, mean=0, std=0.1)
        
        
        return {"timeseries" : timeseries.astype(np.float32), "label" : label, "subjId" : subjId}


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np
import random
from copy import deepcopy

class TripletDataset(Dataset):
    def __init__(self, datasetDetails):
        self.dataset = SupervisedDataset(datasetDetails)
        self.batchSize = datasetDetails.batchSize
        self.dynamicLength = datasetDetails.dynamicLength
        self.foldCount = datasetDetails.foldCount
        self.seed = datasetDetails.datasetSeed
        self.num_triplets = datasetDetails.numTriplets

        # Load both modalities
        self.fmri_data, self.fmri_labels, self.fmri_subjectIds = hcpWorkingMemLoader(datasetDetails.atlas, datasetDetails.targetTask)
        self.fnirs_data, self.fnirs_labels, self.fnirs_subjectIds = hcpfNIRSLoader(datasetDetails.atlas, datasetDetails.targetTask, datasetDetails.signal)

        # Label modality (0 for fMRI, 1 for fNIRS)
        self.fmri_modality = [0] * len(self.fmri_data)
        self.fnirs_modality = [1] * len(self.fnirs_data)

        # Combine data for fold separation
        self.data = self.fmri_data + self.fnirs_data
        self.labels = self.fmri_labels + self.fnirs_labels
        self.subjectIds = self.fmri_subjectIds + self.fnirs_subjectIds
        self.modalities = self.fmri_modality + self.fnirs_modality
        self.groups = self.subjectIds  # used for GroupKFold

        # KFold setup
        self.kFold = GroupKFold(self.foldCount) if self.foldCount else None
        self.triplets = []

    def __len__(self):
        return len(self.triplets)

    def mine_triplets(self):
        """Mine triplets using only self.targetData and self.targetLabels for the current fold."""
        triplets = []

        # Separate the data by modality from the current fold
        anchor_pool = [(i, d, l) for i, (d, l, m) in enumerate(zip(self.targetData, self.targetLabels, self.targetModalities)) if m == 0]  # fMRI
        positive_pool = [(i, d, l) for i, (d, l, m) in enumerate(zip(self.targetData, self.targetLabels, self.targetModalities)) if m == 1]  # fNIRS

        label_to_fmri = {}
        label_to_fnirs = {}

        for _, data, label in anchor_pool:
            label_to_fmri.setdefault(label, []).append(data)
        for _, data, label in positive_pool:
            label_to_fnirs.setdefault(label, []).append(data)

        labels = list(set(label_to_fmri.keys()) & set(label_to_fnirs.keys()))
        rng = random.Random(self.seed)

        for _ in range(self.num_triplets):
            label = rng.choice(labels)
            anchor = rng.choice(label_to_fmri[label])
            positive = rng.choice(label_to_fnirs[label])

            # Pick negative label
            negative_label = rng.choice([l for l in labels if l != label])
            # Choose randomly negative from fMRI or fNIRS
            if rng.random() < 0.5:
                negative = rng.choice(label_to_fmri[negative_label])
            else:
                negative = rng.choice(label_to_fnirs[negative_label])

            triplets.append((anchor, positive, negative))

        self.triplets = triplets

    def setFold(self, fold, train=True):
        self.k = fold
        self.train = train

        if self.foldCount is None:
            raise ValueError("Fold count is required for triplet training with GroupKFold.")
        
        splits = list(self.kFold.split(self.data, self.labels, self.groups))
        trainIdx, testIdx = splits[fold]

        train_groups = set(self.groups[idx] for idx in trainIdx)
        test_groups = set(self.groups[idx] for idx in testIdx)
        intersect = train_groups.intersection(test_groups)
        if intersect:
            raise ValueError(f"Group(s) {intersect} found in both train and test splits!")

        idx_to_use = trainIdx if train else testIdx
        self.targetData = [self.data[i] for i in idx_to_use]
        self.targetLabels = [self.labels[i] for i in idx_to_use]
        self.targetSubjIds = [self.subjectIds[i] for i in idx_to_use]
        self.targetModalities = [self.modalities[i] for i in idx_to_use]

        if train:
            self.mine_triplets()  # Only mine triplets for training
        else:
            self.triplets = []  # No triplets needed for test mode

    def getFold(self, fold, train=True):
        dataset_copy = deepcopy(self)
        dataset_copy.setFold(fold, train=train)
        return DataLoader(dataset_copy, batch_size=dataset_copy.batchSize if train else 1, shuffle=False)

        





