import os
import numpy as np
import sys
import scipy.io
import torch
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils
import prep_atlas
# Define the path
_, MNI = fnirs_utils.load_fnirs(os.path.join(os.path.dirname(__file__), 'fNIRS'))

# Load the atlas
atlas_img = prep_atlas.prep_atlas("AAL")

fnirs_utils.get_parcel_label(MNI[0],atlas_img.get_fdata(), atlas_img.affine)