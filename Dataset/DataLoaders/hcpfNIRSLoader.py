import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils
datadir = "./Dataset/Data/fNIRS"


def healthCheckOnRoiSignal(roiSignal):
    """
        roiSignal : (N, T)
    """


    # remove subjects with dead rois
    if(np.sum(np.sum(np.abs(roiSignal), axis=1) == 0) > 0):
        return False

    return True    

def hcpfNIRSLoader(atlas, targetTask):

    """
        x : (#subjects, N)
    """
    signal = 'HbT'
    invert = True
    dataset = fnirs_utils.load_fnirs_subject(1, 'nback', signal)

    x = []
    y = []
    subjectIds = []
   
    for data in dataset:
        
        label = int(data["pheno"]["label"])
        
        

        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):
            if invert:
                data["roiTimeseries"] =  -1 * data["roiTimeseries"]
           
            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"][1]))
        else:
            print("Skipping subject: ", data["pheno"]["subjectId"])

    return x, y, subjectIds
