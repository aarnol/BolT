import os
import numpy as np
import sys
import scipy.io
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils
# Define the path
# fnirs_data, MNI = fnirs_utils.load_fnirs(os.path.join(os.path.dirname(__file__), 'fNIRS'))

data = torch.load(os.path.join('Dataset\Data\dataset_hcp_AAL_fNIRS_5.save'))
print(data[0]['roiTimeseries'][0].shape)
