import torch
import numpy as np
import sys
import os
datadir = "./Dataset/Data"
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
from fnirs_utils import getBadChannels
fnirs_dir = "./Dataset/Data/fNIRS/fNIRS28-1.38"
bad_channels = getBadChannels(fnirs_dir)
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

    dataset = torch.load(datadir + "/SFN_data/hcpMotor_sphere_6_MNI30.save", weights_only=False)

    x = []
    y = []
    subjectIds = []
    
    
        
    for data in dataset:
    
        label = int(data["pheno"]["label"])
        #only get left vs right hand
        if label == 3:
            label = 0
        elif label != 1:
            continue
        #filter out bad channels
        #data["roiTimeseries"] = np.delete(data["roiTimeseries"], bad_channels, axis=1)
    
        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):
            if(data['roiTimeseries'].shape[0] ==8):
                print("Skipping subject: ", data["pheno"]["subjectId"])
                continue
            
            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            print("Skipping subject: ", data["pheno"]["subjectId"])
            

       

    return x, y, subjectIds
