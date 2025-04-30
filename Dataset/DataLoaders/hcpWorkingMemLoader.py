import torch
import numpy as np

datadir = "./Dataset/Data"
bad_channels = np.loadtxt('./bad_channel_indices.txt', dtype=int)

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

    dataset = torch.load(datadir + "/hcpWM_sphere30_sphere107.save")

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
