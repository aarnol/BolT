import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils
datadir = "./Dataset/Data/fNIRS/fNIRS28-1.38/"


def healthCheckOnRoiSignal(roiSignal):
    """
        roiSignal : (N, T)
    """


    # remove subjects with dead rois
    if(np.sum(np.sum(np.abs(roiSignal), axis=1) == 0) > 0):
        return False

    return True    

def hcpfNIRSLoader(atlas, targetTask, signal, subject= None):

    """
        x : (#subjects, N)
    """
    
    dataset = fnirs_utils.load28(datadir, type = signal)

    x = []
    y = []
    subjectIds = []
   
    for data in dataset:
        
        label = int(data["pheno"]["label"])
        
        
        

        if(healthCheckOnRoiSignal(data["roiTimeseries"].T) or 1):
            
            
            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            print("Skipping subject: ", data["pheno"]["subjectId"], data["pheno"]["label"])

    return x, y, subjectIds
