import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils



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
    
    dataset = fnirs_utils.get_task_data(signal, targetTask)
    bad_channels = fnirs_utils.get_bad_channels()
    

    x = []
    y = []
    subjectIds = []
   
    for data in dataset:
        
        label = int(data["pheno"]["label"])
        if targetTask != "nback":
            if label == 3:
                label = 0
            elif label != 1:
                continue

        
        

        if(healthCheckOnRoiSignal(data["roiTimeseries"].T) or 1):
            
            ts = data["roiTimeseries"].T
            # bad channels are already removed in fnirs_utils.get_task_data
            print(ts.shape)
            x.append(ts)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            print("Skipping subject: ", data["pheno"]["subjectId"], data["pheno"]["label"])
        

    return x, y, subjectIds
