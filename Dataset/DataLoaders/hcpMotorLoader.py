import torch
import numpy as np

datadir = "./Dataset/Data"

only_fingers = True
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

    dataset = torch.load(datadir + "/hcp_motor_{}_15.save".format(atlas))

    x = []
    y = []
    subjectIds = []
    
    for data in dataset:
        
        label = int(data["pheno"]["label"])
        if only_fingers and label == 3:
            label = 0
        elif only_fingers and label!=1:
            continue
            

        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):

            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            print("Skipping subject: ", data["pheno"]["subjectId"])

    return x, y, subjectIds
