import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
from fnirs_utils import get_bad_channels
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

def hcpMotorLoader(atlas, targetTask):

    """
        x : (#subjects, N)
    """
    
    dataset = torch.load(datadir + "/final/hcpMotor_sphere_new_mni_30.save")
    bad_channels = get_bad_channels()
    
  
    x = []
    y = []
    subjectIds = []
    
    for data in dataset:
        
        label = int(data["pheno"]["label"])
        if label == 3:
            label = 0
        elif label != 1:
            continue
        
        #filter out bad channels and non PFC
        # with open("channel_regions.txt", "r") as f:
        #     channel_regions = [line.strip() for line in f if line.strip()]

        # prefrontal_channels = np.where(np.char.find(np.char.lower(channel_regions), "prefrontal") != -1)[0]
        # Get indices that are both prefrontal and not bad channels
        # valid_channels = np.setdiff1d(prefrontal_channels, bad_channels)
        print("Original shape:", data["roiTimeseries"].shape)
        print("Bad channels to remove:", bad_channels)
        new= np.delete(data["roiTimeseries"], bad_channels, axis=1)
        print("Shape after removing bad channels:", new.shape)
        
        ts = new.T.squeeze()
        if(healthCheckOnRoiSignal(ts)):
            if(data['roiTimeseries'].shape[0] ==8):
                print("Skipping subject: ", data["pheno"]["subjectId"])
                continue
            x.append(ts)
            print(ts.shape)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            print("Skipping subject: ", data["pheno"]["subjectId"])

    #print distribution of labels
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution: ", dict(zip(unique, counts)))
    print('fmri shape:', x[0].shape)
    return x, y, subjectIds