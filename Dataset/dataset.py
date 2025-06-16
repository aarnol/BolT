from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import GroupKFold
from random import shuffle, randrange
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from .DataLoaders.hcpWorkingMemLoader import hcpWorkingMemLoader
from .DataLoaders.hcpfNIRSLoader import hcpfNIRSLoader
from .DataLoaders.hcpMotorLoader import hcpMotorLoader
import os
from copy import deepcopy
import torch.nn.functional as F

loaderMapper = {
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

def online_triplet_collate_fn(batch):
    """
    Custom collate function for online triplet mining.
    Returns padded timeseries, labels, modalities, and subject IDs.
    """
    timeseries = [torch.tensor(item['timeseries']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    modalities = torch.tensor([item['modality'] for item in batch])
    subj_ids = [item['subjId'] for item in batch]
    
    # Find the maximum time dimension in the batch
    max_time = max(seq.size(1) for seq in timeseries)
    
    # Pad or truncate all sequences to the same time dimension
    aligned_inputs = []
    for seq in timeseries:
        if seq.size(1) < max_time:  # Pad if shorter
            padding = torch.zeros(seq.size(0), max_time - seq.size(1))
            aligned_inputs.append(torch.cat([seq, padding], dim=1))
        elif seq.size(1) > max_time:  # Truncate if longer
            aligned_inputs.append(seq[:, :max_time])
        else:  # Already aligned
            aligned_inputs.append(seq)
    
    # Stack aligned inputs
    timeseries_padded = torch.stack(aligned_inputs)
    
    return {
        'timeseries': timeseries_padded,
        'labels': labels,
        'modalities': modalities,
        'subj_ids': subj_ids
    }
from torch.utils.data import Sampler
import random
from collections import defaultdict



def getDataset(options):
    return SupervisedDataset(options)

def getTripletDataset(options):
    return OnlineTripletDataset(options)
def getContrastiveDataset(options):
    return OnlineContrastiveDataset(options)
def getBalancedContrastiveDataset(options):
    return BalancedContrastiveDataset(options)

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
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict
import random

from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict
import random

class BalancedModalityBatchSampler(Sampler):
    """
    Ensures each batch has a balanced number of samples from each modality.
    fMRI (modality 0): sampled without replacement.
    fNIRS (modality 1): sampled with replacement.
    """
    def __init__(self, dataset, batch_size, ratio=0.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ratio = ratio  # Proportion of fMRI

        # Group indices by modality
        self.indices_by_modality = defaultdict(list)
        for idx, modality in enumerate(dataset.targetModalities):
            self.indices_by_modality[modality].append(idx)

        self.fmri_indices = self.indices_by_modality[0]
        self.fnirs_indices = self.indices_by_modality[1]

        self.fmri_len = len(self.fmri_indices)
        self.fnirs_len = len(self.fnirs_indices)

        self.fmri_batch_size = int(self.batch_size * self.ratio)
        self.fnirs_batch_size = self.batch_size - self.fmri_batch_size

        # Compute how many batches we can make with fmri data (since it's without replacement)
        self.num_batches = self.fmri_len // self.fmri_batch_size

    def __iter__(self):
        fmri_pool = self.fmri_indices.copy()
        random.shuffle(fmri_pool)

        for i in range(self.num_batches):
            fmri_batch = fmri_pool[i * self.fmri_batch_size:(i + 1) * self.fmri_batch_size]
            fnirs_batch = random.choices(self.fnirs_indices, k=self.fnirs_batch_size)  # with replacement
            batch = fmri_batch + fnirs_batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


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
            #timeseries = guassianNoise(timeseries, mean=0, std=0.1)
        
        
        return {"timeseries" : timeseries.astype(np.float32), "label" : label, "subjId" : subjId}


class OnlineTripletDataset(Dataset):
    """
    Dataset for online triplet mining similar to FaceNet.
    Instead of pre-mining triplets, we return individual samples and 
    mine triplets dynamically from each batch during training.
    """
    
    def __init__(self, datasetDetails):
        self.batchSize = datasetDetails.batchSize
        self.dynamicLength = datasetDetails.dynamicLength
        self.foldCount = datasetDetails.foldCount
        self.seed = datasetDetails.datasetSeed
        
        # Load both modalities
        self.fmri_data, self.fmri_labels, self.fmri_subjectIds = hcpWorkingMemLoader(
            datasetDetails.atlas, datasetDetails.targetTask
        )
        self.fnirs_data, self.fnirs_labels, self.fnirs_subjectIds = hcpfNIRSLoader(
            datasetDetails.atlas, datasetDetails.targetTask, datasetDetails.signal
        )
        
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
        
        # Target data for current fold
        self.targetData = None
        self.targetLabels = None
        self.targetSubjIds = None
        self.targetModalities = None
        self.randomRanges = None
        self.train = None

    def __len__(self):
        return len(self.targetData) if self.targetData is not None else len(self.data)
    
    def get_nOfTrains_perFold(self):
        if self.foldCount is not None:
            return int(np.ceil(len(self.data) * (self.foldCount - 1) / self.foldCount)*2.5)
        else:
            return len(self.data)

    def setFold(self, fold, train=True):
        """Set the current fold for training or testing."""
        self.k = fold
        self.train = train

        if self.foldCount is None:
            raise ValueError("Fold count is required for triplet training with GroupKFold.")
        
        splits = list(self.kFold.split(self.data, self.labels, self.groups))
        trainIdx, testIdx = splits[fold]
        
        # Save the groups for that fold to a file for later use
        with open(f"fold_{fold}_groups.txt", "w") as f:
            f.write(f"Train groups: {set(self.groups[idx] for idx in trainIdx)}\n")
            f.write(f"Test groups: {set(self.groups[idx] for idx in testIdx)}\n")
        
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

        if train and self.dynamicLength is not None:
            np.random.seed(self.seed + 1)
            self.randomRanges = [
                [np.random.randint(0, self.targetData[i].shape[-1] - self.dynamicLength) for _ in range(9999)]
                for i in range(len(self.targetData))
            ]

    def getFold(self, fold, train=True):
        """Get a DataLoader for a specific fold."""
        dataset_copy = deepcopy(self)
        dataset_copy.setFold(fold, train=train)
        batch_sampler = BalancedModalityBatchSampler(dataset_copy, dataset_copy.batchSize, ratio=0.5)
        return DataLoader(
            dataset_copy, 
            #batch_size=dataset_copy.batchSize, 
            # shuffle=train,  # Shuffle for training
            batch_sampler=batch_sampler,  # Use custom sampler for training
            collate_fn=online_triplet_collate_fn
        )

    def __getitem__(self, idx):
        """Return a single sample with timeseries, label, modality, and subject ID."""
        timeseries = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]
        modality = self.targetModalities[idx]

        # Normalize timeseries (z-score)
        timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1, keepdims=True)
        timeseries = np.nan_to_num(timeseries, 0)
        
        # Dynamic sampling if training
        if self.train and self.dynamicLength is not None:
            samplingInit = self.randomRanges[idx].pop()
            timeseries = timeseries[:, samplingInit:samplingInit + self.dynamicLength]
            
            # Optionally add noise during training
            if modality == 1:  # fNIRS
                timeseries = guassianNoise(timeseries, mean=0, std=0.1)
        
        return {
            "timeseries": timeseries.astype(np.float32),
            "label": label,
            "modality": modality,
            "subjId": subjId
        }


# Triplet mining functions for use during training
def get_triplet_mask(labels, modalities):
    """
    Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
    - i, j, k are distinct
    - labels[i] == labels[j] and labels[i] != labels[k]
    - modalities[i] != modalities[j] (cross-modal positive)
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    
    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    
    valid_labels = i_equal_j & (~i_equal_k)
    
    # Check if modalities[i] != modalities[j] (cross-modal positive)
    modality_equal = modalities.unsqueeze(0) == modalities.unsqueeze(1)
    i_not_equal_j_modality = (~modality_equal).unsqueeze(2)
    
    return distinct_indices & valid_labels & i_not_equal_j_modality


def batch_all_triplet_loss(embeddings, labels, modalities, margin=1.0):
    """
    Build the triplet loss over a batch of embeddings using online mining.
    
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        labels: tensor of shape (batch_size,)
        modalities: tensor of shape (batch_size,)
        margin: margin for triplet loss
        
    Returns:
        triplet_loss: average triplet loss over valid triplets
    """
    # Get the pairwise distance matrix
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    
    # Get anchor-positive distances and anchor-negative distances
    anchor_positive_dist = pairwise_dist.unsqueeze(2)  # [batch, batch, 1]
    anchor_negative_dist = pairwise_dist.unsqueeze(1)  # [batch, 1, batch]
    
    # Compute triplet loss: max(d(a,p) - d(a,n) + margin, 0)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    
    # Put to zero the invalid triplets
    mask = get_triplet_mask(labels, modalities)
    triplet_loss = mask.float() * triplet_loss
    
    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.clamp(triplet_loss, min=0.0)
    
    # Count number of positive triplets (where loss > 0)
    valid_triplets = triplet_loss > 1e-16
    num_positive_triplets = valid_triplets.sum()
    
    # Get final mean triplet loss over the positive valid triplets
    if num_positive_triplets > 0:
        triplet_loss = triplet_loss.sum() / num_positive_triplets.float()
    else:
        triplet_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    return triplet_loss


def batch_hard_triplet_loss(embeddings, labels, modalities, margin=1.0):
    """
    Build the triplet loss over a batch of embeddings using hard mining.
    For each anchor, we select the hardest positive and hardest negative.
    
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        labels: tensor of shape (batch_size,)
        modalities: tensor of shape (batch_size,)
        margin: margin for triplet loss
        
    Returns:
        triplet_loss: average triplet loss over valid anchors
    """
    # Get the pairwise distance matrix
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    
    # For each anchor, get the hardest positive
    # First, create mask for valid positives (same label, different modality)
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    modality_different = modalities.unsqueeze(0) != modalities.unsqueeze(1)
    valid_positives = label_equal & modality_different
    
    # Set distance to negatives and self to a large value so they won't be selected as hardest positive
    masked_pos_dist = pairwise_dist.clone()
    masked_pos_dist[~valid_positives] = float('inf')
    
    # Get hardest positive for each anchor
    hardest_positive_dist, _ = masked_pos_dist.min(dim=1)
    
    # For each anchor, get the hardest negative
    # First, create mask for valid negatives (different label)
    valid_negatives = ~label_equal
    
    # Set distance to positives and self to a small value so they won't be selected as hardest negative
    masked_neg_dist = pairwise_dist.clone()
    masked_neg_dist[~valid_negatives] = 0.0
    
    # Get hardest negative for each anchor
    hardest_negative_dist, _ = masked_neg_dist.max(dim=1)
    
    # Compute triplet loss for each valid anchor
    triplet_loss = hardest_positive_dist - hardest_negative_dist + margin
    triplet_loss = torch.clamp(triplet_loss, min=0.0)
    
    # Only consider anchors that have valid positives
    valid_anchors = (hardest_positive_dist != float('inf'))
    
    if valid_anchors.sum() > 0:
        triplet_loss = triplet_loss[valid_anchors].mean()
    else:
        triplet_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    return triplet_loss

class OnlineContrastiveDataset(Dataset):
    """
    Dataset for online contrastive learning.
    Returns pairs of samples with similarity labels (1 for same class, 0 for different class).
    """
    
    def __init__(self, datasetDetails):
        self.batchSize = datasetDetails.batchSize
        self.dynamicLength = datasetDetails.dynamicLength
        self.foldCount = datasetDetails.foldCount
        self.seed = datasetDetails.datasetSeed
        
        # Load both modalities
        self.fmri_data, self.fmri_labels, self.fmri_subjectIds = hcpWorkingMemLoader(
            datasetDetails.atlas, datasetDetails.targetTask
        )
        self.fnirs_data, self.fnirs_labels, self.fnirs_subjectIds = hcpfNIRSLoader(
            datasetDetails.atlas, datasetDetails.targetTask, datasetDetails.signal
        )
        
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
        
        # Target data for current fold
        self.targetData = None
        self.targetLabels = None
        self.targetSubjIds = None
        self.targetModalities = None
        self.randomRanges = None
        self.train = None
        
        # For contrastive learning
        self.pairs = None
        self.pair_labels = None

    def __len__(self):
        return len(self.pairs) if self.pairs is not None else 0
    
    def get_nOfTrains_perFold(self):
        if self.foldCount is not None:
            # Estimate based on number of possible pairs
            n_samples = int(np.ceil(len(self.data) * (self.foldCount - 1) / self.foldCount))
            return n_samples * 10  # Approximate number of pairs to generate
        else:
            return len(self.data) * 10

    def _create_pairs(self, positive_ratio=0.5):
        """Create pairs of samples for contrastive learning."""
        if self.targetData is None:
            raise ValueError("Must call setFold() first")
        print("Creating pairs for contrastive learning...")
        np.random.seed(self.seed + self.k if hasattr(self, 'k') else self.seed)
        
        n_samples = len(self.targetData)
        pairs = []
        pair_labels = []
        
        # Create label-to-indices mapping for efficient positive pair generation
        label_to_indices = {}
        for idx, label in enumerate(self.targetLabels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        # Calculate number of positive and negative pairs
        n_pairs_per_epoch = n_samples * 2  # Generate 2x samples as pairs
        n_positive = int(n_pairs_per_epoch * positive_ratio)
        n_negative = n_pairs_per_epoch - n_positive
        
        # Generate positive pairs (same class)
        for _ in range(n_positive):
            # Choose a random label that has at least 2 samples
            valid_labels = [label for label, indices in label_to_indices.items() if len(indices) >= 2]
            if not valid_labels:
                break
            
            label = np.random.choice(valid_labels)
            idx1, idx2 = np.random.choice(label_to_indices[label], 2, replace=False)
            pairs.append((idx1, idx2))
            pair_labels.append(1)  # Same class
        
        # Generate negative pairs (different class)
        for _ in range(n_negative):
            idx1 = np.random.randint(0, n_samples)
            # Find samples with different labels
            different_label_indices = [i for i, label in enumerate(self.targetLabels) 
                                     if label != self.targetLabels[idx1]]
            if different_label_indices:
                idx2 = np.random.choice(different_label_indices)
                pairs.append((idx1, idx2))
                pair_labels.append(0)  # Different class
        
        self.pairs = pairs
        self.pair_labels = pair_labels
        print(f"Created {len(pairs)} pairs: {len(pair_labels)} positive and {len(pair_labels) - sum(pair_labels)} negative.")

    def setFold(self, fold, train=True):
        """Set the current fold for training or testing."""
        self.k = fold
        self.train = train

        if self.foldCount is None:
            raise ValueError("Fold count is required for contrastive training with GroupKFold.")
        
        splits = list(self.kFold.split(self.data, self.labels, self.groups))
        trainIdx, testIdx = splits[fold]
        
        # Save the groups for that fold to a file for later use
        with open(f"fold_{fold}_groups.txt", "w") as f:
            f.write(f"Train groups: {set(self.groups[idx] for idx in trainIdx)}\n")
            f.write(f"Test groups: {set(self.groups[idx] for idx in testIdx)}\n")
        
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

        if train and self.dynamicLength is not None:
            np.random.seed(self.seed + 1)
            self.randomRanges = [
                [np.random.randint(0, self.targetData[i].shape[-1] - self.dynamicLength) for _ in range(9999)]
                for i in range(len(self.targetData))
            ]
        
        # Create pairs for contrastive learning
        self._create_pairs()

    def getFold(self, fold, train=True):
        """Get a DataLoader for a specific fold."""
        dataset_copy = deepcopy(self)
        dataset_copy.setFold(fold, train=train)
        return DataLoader(
            dataset_copy, 
            batch_size=dataset_copy.batchSize, 
            shuffle=train,  # Shuffle for training
            collate_fn=contrastive_collate_fn
        )

    def _get_processed_sample(self, idx):
        """Get a processed sample by index."""
        timeseries = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]
        modality = self.targetModalities[idx]

        # Normalize timeseries (z-score)
        timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1, keepdims=True)
        timeseries = np.nan_to_num(timeseries, 0)
        
        # Dynamic sampling if training
        if self.train and self.dynamicLength is not None:
            samplingInit = self.randomRanges[idx].pop()
            timeseries = timeseries[:, samplingInit:samplingInit + self.dynamicLength]
            
            # Optionally add noise during training
            if modality == 1:  # fNIRS
                timeseries = guassianNoise(timeseries, mean=0, std=0.1)
        
        return {
            "timeseries": timeseries.astype(np.float32),
            "label": label,
            "modality": modality,
            "subjId": subjId
        }

    def __getitem__(self, idx):
        """Return a pair of samples with similarity label."""
        idx1, idx2 = self.pairs[idx]
        similarity_label = self.pair_labels[idx]
        
        sample1 = self._get_processed_sample(idx1)
        sample2 = self._get_processed_sample(idx2)
        print()
        return {
            "sample1": sample1,
            "sample2": sample2,
            "similarity": similarity_label
        }


def contrastive_collate_fn(batch):
    """
    Collate function for contrastive learning batches.
    """
    sample1_batch = {
        "timeseries": torch.stack([torch.tensor(item["sample1"]["timeseries"]) for item in batch]),
        "label": torch.tensor([item["sample1"]["label"] for item in batch]),
        "modality": torch.tensor([item["sample1"]["modality"] for item in batch]),
        "subjId": [item["sample1"]["subjId"] for item in batch]
    }
    
    sample2_batch = {
        "timeseries": torch.stack([torch.tensor(item["sample2"]["timeseries"]) for item in batch]),
        "label": torch.tensor([item["sample2"]["label"] for item in batch]),
        "modality": torch.tensor([item["sample2"]["modality"] for item in batch]),
        "subjId": [item["sample2"]["subjId"] for item in batch]
    }
    
    similarities = torch.tensor([item["similarity"] for item in batch], dtype=torch.float32)
    
    return {
        "sample1": sample1_batch,
        "sample2": sample2_batch,
        "similarities": similarities
    }


# Recommended: Balanced Contrastive Dataset for imbalanced modality data
class BalancedContrastiveDataset(OnlineContrastiveDataset):
    """
    Contrastive dataset optimized for imbalanced modality data.
    Ensures balanced representation of cross-modal and within-modal pairs
    regardless of the original modality distribution.
    """
    
    def _create_pairs(self, positive_ratio=0.5, cross_modal_ratio=0.7):
        """Create balanced pairs including cross-modal considerations."""
        if self.targetData is None:
            raise ValueError("Must call setFold() first")
        
        np.random.seed(self.seed + self.k if hasattr(self, 'k') else self.seed)
        
        n_samples = len(self.targetData)
        pairs = []
        pair_labels = []
        
        # Separate indices by modality and label
        fmri_indices = [i for i, mod in enumerate(self.targetModalities) if mod == 0]
        fnirs_indices = [i for i, mod in enumerate(self.targetModalities) if mod == 1]
        
        # Create label-to-indices mapping for each modality
        fmri_label_to_indices = {}
        fnirs_label_to_indices = {}
        
        for idx in fmri_indices:
            label = self.targetLabels[idx]
            if label not in fmri_label_to_indices:
                fmri_label_to_indices[label] = []
            fmri_label_to_indices[label].append(idx)
        
        for idx in fnirs_indices:
            label = self.targetLabels[idx]
            if label not in fnirs_label_to_indices:
                fnirs_label_to_indices[label] = []
            fnirs_label_to_indices[label].append(idx)
        
        n_pairs_per_epoch = n_samples * 2
        n_positive = int(n_pairs_per_epoch * positive_ratio)
        n_negative = n_pairs_per_epoch - n_positive
        
        # For positive pairs
        n_cross_modal_pos = int(n_positive * cross_modal_ratio)
        n_within_modal_pos = n_positive - n_cross_modal_pos
        
        # Generate cross-modal positive pairs (same label, different modality)
        common_labels = set(fmri_label_to_indices.keys()) & set(fnirs_label_to_indices.keys())
        cross_modal_pos_count = 0
        # Try to generate all requested cross-modal positive pairs
        for _ in range(n_cross_modal_pos * 3):  # Allow more attempts for imbalanced data
            if not common_labels or cross_modal_pos_count >= n_cross_modal_pos:
                break
            label = np.random.choice(list(common_labels))
            if (fmri_label_to_indices[label] and fnirs_label_to_indices[label]):
                idx1 = np.random.choice(fmri_label_to_indices[label])
                idx2 = np.random.choice(fnirs_label_to_indices[label])
                pairs.append((idx1, idx2))
                pair_labels.append(1)
                cross_modal_pos_count += 1
        
        # Generate within-modal positive pairs (fill remaining positive pairs)
        within_modal_pos_count = 0
        remaining_pos = n_positive - cross_modal_pos_count
        
        for _ in range(remaining_pos * 2):  # Allow attempts
            if within_modal_pos_count >= remaining_pos:
                break
                
            # Prioritize the minority modality to balance representation
            minority_is_fmri = len(fmri_indices) < len(fnirs_indices)
            use_fmri = (np.random.random() < 0.7 if minority_is_fmri else np.random.random() < 0.3)
            
            if use_fmri and fmri_label_to_indices:
                # fMRI within-modal
                valid_labels = [label for label, indices in fmri_label_to_indices.items() if len(indices) >= 2]
                if valid_labels:
                    label = np.random.choice(valid_labels)
                    idx1, idx2 = np.random.choice(fmri_label_to_indices[label], 2, replace=False)
                    pairs.append((idx1, idx2))
                    pair_labels.append(1)
                    within_modal_pos_count += 1
            elif fnirs_label_to_indices:
                # fNIRS within-modal
                valid_labels = [label for label, indices in fnirs_label_to_indices.items() if len(indices) >= 2]
                if valid_labels:
                    label = np.random.choice(valid_labels)
                    idx1, idx2 = np.random.choice(fnirs_label_to_indices[label], 2, replace=False)
                    pairs.append((idx1, idx2))
                    pair_labels.append(1)
                    within_modal_pos_count += 1
        
        # Generate negative pairs (different labels)
        n_cross_modal_neg = int(n_negative * cross_modal_ratio)
        n_within_modal_neg = n_negative - n_cross_modal_neg
        
        # Cross-modal negative pairs
        for _ in range(n_cross_modal_neg):
            if fmri_indices and fnirs_indices:
                idx1 = np.random.choice(fmri_indices)
                # Find fNIRS samples with different labels
                different_label_indices = [i for i in fnirs_indices 
                                         if self.targetLabels[i] != self.targetLabels[idx1]]
                if different_label_indices:
                    idx2 = np.random.choice(different_label_indices)
                    pairs.append((idx1, idx2))
                    pair_labels.append(0)
        
        # Within-modal negative pairs
        for _ in range(n_within_modal_neg):
            if np.random.random() < 0.5 and len(fmri_indices) >= 2:
                # fMRI within-modal negative
                idx1 = np.random.choice(fmri_indices)
                different_label_indices = [i for i in fmri_indices 
                                         if self.targetLabels[i] != self.targetLabels[idx1]]
                if different_label_indices:
                    idx2 = np.random.choice(different_label_indices)
                    pairs.append((idx1, idx2))
                    pair_labels.append(0)
            elif len(fnirs_indices) >= 2:
                # fNIRS within-modal negative
                idx1 = np.random.choice(fnirs_indices)
                different_label_indices = [i for i in fnirs_indices 
                                         if self.targetLabels[i] != self.targetLabels[idx1]]
                if different_label_indices:
                    idx2 = np.random.choice(different_label_indices)
                    pairs.append((idx1, idx2))
                    pair_labels.append(0)
        
        self.pairs = pairs
        self.pair_labels = pair_labels