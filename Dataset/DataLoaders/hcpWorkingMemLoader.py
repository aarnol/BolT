import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
from fnirs_utils import getBadChannels
datadir = "./Dataset/Data"
fnirs_dir = "./Dataset/Data/fNIRS/fNIRS28-1.38"


def healthCheckOnRoiSignal(roiSignal):
    """
        roiSignal : (N, T)
    """


    # remove subjects with dead rois
    if(np.sum(np.sum(np.abs(roiSignal), axis=1) == 0) > 0):
        return False

    return True    

def hcpWorkingMemLoader(atlas, targetTask):

    """
        x : (#subjects, N)
    """
    
    dataset = torch.load(datadir + "/hcpWM_sphere_newMNI.save")
    bad_channels = getBadChannels(fnirs_dir)
    x = []
    y = []
    subjectIds = []
    
    for data in dataset:
        
        label = int(data["pheno"]["label"])
        
        #filter out bad channels
        data["roiTimeseries"] = np.delete(data["roiTimeseries"], bad_channels, axis=1)
       
        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):
            if(data['roiTimeseries'].shape[0] ==8):
                print("Skipping subject: ", data["pheno"]["subjectId"])
                continue
            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            print("Skipping subject: ", data["pheno"]["subjectId"])

    #print distribution of labels
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution: ", dict(zip(unique, counts)))
    return x, y, subjectIds


import torch
import numpy as np
from itertools import permutations, islice
from collections import defaultdict

def hcpWorkingMemLoaderConcatenatePermuted(atlas, target_task, max_perms_per_group=10):
    """
    Collects fNIRS/fMRI samples grouped by subject and label,
    and concatenates them in different permutations (using only 3 samples per permutation).

    Returns:
        x_concat: list of np.arrays, each (channels, total_time)
        y_concat: list of int labels
        subject_ids_concat: list of subject IDs
    """
    dataset = torch.load(datadir + "/hcpWM_sphere_newMNI.save")
    bad_channels = getBadChannels(fnirs_dir)
    grouped_data = defaultdict(list)  # {(subjectId, label): [samples]}

    for data in dataset:
        label = int(data["pheno"]["label"])
        subj_id = int(data["pheno"]["subjectId"])

        roi_ts = data["roiTimeseries"]
        roi_ts = np.delete(roi_ts, bad_channels, axis=1)

        if healthCheckOnRoiSignal(roi_ts.T):
            grouped_data[(subj_id, label)].append(roi_ts.T)
        else:
            print("Skipping subject:", subj_id)

    x_concat = []
    y_concat = []
    subject_ids_concat = []

    for (subj_id, label), samples in grouped_data.items():
        if len(samples) < 3:
            continue  # Need at least 3 to create permutations

        for perm in islice(permutations(samples, 3), max_perms_per_group):
            concatenated = np.concatenate(perm, axis=1)  # (channels, total_time)
            #z score so each channel has mean 0 and std 1
            concatenated = (concatenated - np.mean(concatenated, axis=1, keepdims=True)) / np.std(concatenated, axis=1, keepdims=True)
            print(concatenated.shape)
            x_concat.append(concatenated)
            y_concat.append(label)
            subject_ids_concat.append(subj_id)

    # Print label distribution
    unique, counts = np.unique(y_concat, return_counts=True)
    print("Label distribution after permutation:", dict(zip(unique, counts)))

    return x_concat, y_concat, subject_ids_concat


#hcpWorkingMemLoader = hcpWorkingMemLoaderConcatenatePermuted
if __name__ == "__main__":
    # Example usage
    
    
    x_concat, y_concat, subject_ids_concat = hcpWorkingMemLoaderConcatenatePermuted()
    print(f"Loaded {len(x_concat)} concatenated samples.")