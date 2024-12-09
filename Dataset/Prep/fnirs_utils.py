import numpy as np
import os

"""
Folder structure:
/fNIRS
    /S1
        /S1_Data.csv
        /S1_Digitization.csv
        /S1_Timings.csv
    /S2
        /S2_Data.csv
        /S2_Digitization.csv
        /S2_Timings.csv
    ...
"""
def load_fnirs(target_folder):
    """
    Load the fNIRS data from the target folder.
    """
    data = []
    digitization = []
    timings = []
    for folder in os.listdir(target_folder):
        data[folder] = {}
        for file in os.listdir(os.path.join(target_folder, folder)):
            if file.endswith("_Data.csv"):
                data.append(np.loadtxt(os.path.join(target_folder, folder, file), delimiter=","))
            elif file.endswith("_Digitization.csv"):
                digitization.append(np.loadtxt(os.path.join(target_folder, folder, file), delimiter=","))
            elif file.endswith("_Timings.csv"):
                timings.append(np.loadtxt(os.path.join(target_folder, folder, file), delimiter=","))
    return np.array(data), np.array(digitization), np.array(timings)


def calc_MNI_average(digitization):
    """
    Calculate the average of the data in MNI space.
    """
    return np.mean(digitization, axis=0)

def process_fnirs(data, timings):
    """
    Process the fNIRS data.
    """
    pass