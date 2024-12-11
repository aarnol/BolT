import numpy as np
import os
import scipy.io

def load_fnirs(target_folder):
    """
    Load the fNIRS data from the target folder.
    """
    fnirs = {}
    MNI_path = os.path.join(target_folder, 'HCP_MNI.mat')
    digitization =scipy.io.loadmat(MNI_path)['MNI']
    data = os.path.join(target_folder, 'HCP_fNIRS_NBack.mat')
    data = scipy.io.loadmat(data)['Data_fNIRS']
    formatted_data = []
    for subject in data:
        blocks = subject[0]
        blocks = [block[0] for block in blocks]
        i = 1
        for block in blocks:
            f_data = {
                'roiTimeseries': block,
                'pheno': {
                    'subjectId': f'S{i}',
                    'encoding': None,
                    'nback': i,
                    'modality': 'fNIRS'
                }
            }
            formatted_data.append(f_data)
            if i == 1:
                i = 0
            else:
                i = 1
    return formatted_data, digitization

def calc_MNI_average(digitization):
    """
    Calculate the average of the data in MNI space.
    """
    return np.mean(digitization, axis=0)


def get_parcel_label(mni_coord, atlas_data, affine):
    # Convert MNI coordinate to voxel indices
    voxel_indices = np.linalg.inv(affine).dot(np.append(mni_coord, 1))[:3]
    voxel_indices = np.round(voxel_indices).astype(int)
    
    # Extract the label at the voxel indices
    label = atlas_data[tuple(voxel_indices)]
    return label