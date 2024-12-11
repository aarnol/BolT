import os
import numpy as np
import sys
import scipy.io
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils
# Define the path
data = fnirs_utils.load_fnirs(os.path.join(os.path.dirname(__file__), 'fNIRS'))
