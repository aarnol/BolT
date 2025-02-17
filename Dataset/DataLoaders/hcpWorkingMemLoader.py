import torch
import numpy as np

datadir = "./Dataset/Data"


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

    dataset = torch.load(datadir + "/dataset_hcpWM_{}15.save".format(atlas))

    x = []
    y = []
    subjectIds = []
    
    for data in dataset:
        
        label = int(data["pheno"]["label"])
        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):

            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            print("Skipping subject: ", data["pheno"]["subjectId"])

    return x, y, subjectIds
