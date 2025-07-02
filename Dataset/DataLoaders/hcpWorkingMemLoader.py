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
    
    dataset = torch.load(datadir + "/SFN_data/hcpWM_sphere_0_MNI30.save")
    bad_channels = getBadChannels(fnirs_dir)
    x = []
    y = []
    subjectIds = []
    
    for data in dataset:
        
        label = int(data["pheno"]["label"])
        
        #filter out bad channels and non PFC
        with open("channel_regions.txt", "r") as f:
            channel_regions = [line.strip() for line in f if line.strip()]

        prefrontal_channels = np.where(np.char.find(np.char.lower(channel_regions), "prefrontal") != -1)[0]
        # Get indices that are both prefrontal and not bad channels
        valid_channels = np.setdiff1d(prefrontal_channels, bad_channels)
        data["roiTimeseries"] = data["roiTimeseries"][:, valid_channels]

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
    print('fmri shape:', x[0].shape)
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
            continue  # Need at least 3 to concatenate

        for i in range(len(samples) - 2):
            concatenated = np.concatenate(samples[i:i + 3], axis=1)  # (channels, total_time)
            
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