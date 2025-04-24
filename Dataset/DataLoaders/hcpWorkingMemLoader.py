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
        #assign a random label isntead of the one in the dataset
        label = np.random.randint(0, 2)
        #filter out bad channels
        data["roiTimeseries"] = np.delete(data["roiTimeseries"], bad_channels, axis=1)
        print("Shape of roiTimeseries: ", data["roiTimeseries"].shape)
        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):
            
            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            print("Skipping subject: ", data["pheno"]["subjectId"])

    return x, y, subjectIds
