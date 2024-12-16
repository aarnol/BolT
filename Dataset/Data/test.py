import os
import numpy as np
import sys
import scipy.io
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils
# Define the path
fnirs_data, MNI = fnirs_utils.load_fnirs(os.path.join(os.path.dirname(__file__), 'fNIRS'))

data = torch.load(os.path.join('Dataset\Data\dataset_hcpWM_sphere.save'))




print(np.array(data[-1]['pheno']['modality']))
# Discard the fnirs modality data
new_data = [d for d in data if d['pheno']['modality'] != 'fNIRS']
print(np.array(new_data[-1]['pheno']['modality']))
check = [d for d in new_data if d['pheno']['modality'] == 'fNIRS']
print(len(check))
torch.save(new_data, os.path.join('Dataset\Data\dataset_hcpWM_sphere_mod.save'))
