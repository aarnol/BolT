import os
import numpy as np
import sys
import scipy.io
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils
# Define the path
fnirs_data, MNI = fnirs_utils.load_fnirs(os.path.join(os.path.dirname(__file__), 'fNIRS'))
print(fnirs_data)
data = torch.load(os.path.join('Dataset\Data\dataset_hcp_schaefer7_400_fNIRS_1.save'))
data += fnirs_data
print(data)