import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from nilearn import datasets, surface
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from scipy.signal import resample

def downsample(fnirs_data, original, new):
    """
    Downsample fNIRS data from one sampling rate to another.

    Args:
        fnirs_data (np.ndarray): The original fNIRS data of shape (time, channels) or (time,).
        original (float): Original sampling rate (Hz).
        new (float): New sampling rate (Hz), typically lower than original.

    Returns:
        np.ndarray: Downsampled fNIRS data.
    """
    if new > original:
        raise ValueError("New sampling rate must be lower than original sampling rate.")

    # Calculate number of samples after downsampling
    duration = fnirs_data.shape[0] / original  # total time in seconds
    
    new_num_samples = int(np.round(duration * new))

    # Use scipy's resample for Fourier method-based resampling
    downsampled = resample(fnirs_data, new_num_samples, axis=0)

    return downsampled


def load_fnirs(target_folder):
    """
    Load the fNIRS data from the target folder.
    """
    
    MNI_path = os.path.join(target_folder, 'HCP_MNI_final.mat')
    digitization =scipy.io.loadmat(MNI_path)['MNI']
    data = os.path.join(target_folder, 'HCP_fNIRS_NBack.mat')
    data = scipy.io.loadmat(data)['Data_fNIRS']
    formatted_data = []
    # from the hcp protocol order
    labels =     [1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0]
    conditions = [4,1,2,4,1,3,2,3,4,1,2,4,1,3,2,3]
    encoding = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    
    sub = 0
    for subject in data:
        i = 0
        blocks = subject[0]
        blocks = [block[0] for block in blocks]
       
        
        for block in blocks:
            
            label = labels[i]
            f_data = {
                'roiTimeseries': block,
                'pheno': {
                    'subjectId': f'S{sub}',
                    'encoding': encoding[i],
                    'nback': label,
                    "condition": conditions[i],
                    'modality': 'fNIRS'
                }
            }
            
            formatted_data.append(f_data)
            i+=1
        print(f"Subject {sub} loaded")
        sub+=1
    labels = [data['pheno']['nback'] for data in formatted_data]
    print(f"label counts: {np.unique(labels, return_counts=True)}")
    return formatted_data, digitization
def load_mni(target_folder):
    MNI_path = os.path.join(target_folder, 'MNIs_Average.mat')

    digitization =scipy.io.loadmat(MNI_path)['MNIs_Mean']

    return digitization
def load_fnirs_subject_mni(subject_id):
    """
    Load the fNIRS data from the target folder.
    """
    target_folder = f"Dataset/Data/fNIRS"
    MNI_path = os.path.join(target_folder, 'HCP_MNI_final.mat')
    digitization =scipy.io.loadmat(MNI_path)['Data_fNIRS'][subject_id-1][1]
   
    return digitization


import os
import pandas as pd
import numpy as np

def load(root, type='HbR'):
    fnirs_data = []
    bad_path = os.path.join(root, 'bad_channels.csv')
    bad_channels = pd.read_csv(bad_path)
    #get the bad channels where count > 9
    bad_channels = bad_channels[bad_channels['Count'] > 9]['Channel'].tolist()
    
    for root_dir, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.csv') and 'block' in file:
                file_path = os.path.join(root_dir, file)
                data = pd.read_csv(file_path)

                # Filter out short-separation channels (e.g., D32 or higher)
                data = data.loc[:, ~data.columns.str.contains(r'D(?:[3-9][2-9]|[4-9][0-9])', regex=True)]
                

                
                

                
                



                # Extract based on type
                if type.lower() == 'hbr':
                    signal = data.loc[:, data.columns.str.contains('hbr', case=False)]
                elif type.lower() == 'hbo':
                    signal = data.loc[:, data.columns.str.contains('hbo', case=False)]
                elif type.lower() == 'hbt':
                    hbo = data.loc[:, data.columns.str.contains('hbo', case=False)].values
                    hbr = data.loc[:, data.columns.str.contains('hbr', case=False)].values
                    signal = hbo + hbr
                else:
                    raise ValueError("Invalid type. Choose from 'HbR', 'HbO', or 'HbT'.")
                
                drop_bad = True
                if drop_bad:
                    # Make a copy of original column list for consistent indexing
                    original_columns = list(signal.columns)
                    bad_indices = []
                    dropped_column_names = []

                    for channel in bad_channels:
                        pattern = channel[:7].lower()
                        
                        # Find matching columns in the *original* column list
                        matches = [i for i, col in enumerate(original_columns) if pattern in col.lower()]
                        bad_indices.extend(matches)
                        dropped_column_names.extend([original_columns[i] for i in matches])
                        
                        # Now drop from the *current* signal DataFrame
                        signal = signal.loc[:, ~signal.columns.str.contains(pattern, case=False, na=False)]

                # # Remove duplicates and sort for consistency
                # bad_indices = sorted(set(bad_indices))

                # # Final checks
                # print(f"Total bad indices: {len(bad_indices)}")
                # print("Bad indices:", bad_indices)

                # # Save to file
                # with open("bad_channel_indices.txt", "w") as f:
                #     for idx in bad_indices:
                #         f.write(f"{idx}\n")

                


                #conver to numpy array
                signal = signal.values
                # Downsample
                fnirs_hz = 3.8153376573826785
                fmri_hz = 1.39
                signal = downsample(signal, fnirs_hz, fmri_hz)  
                
                # Get the subject ID and label from the file name or directory structure
                dir_name = os.path.basename(root_dir)
                
                label = file[-5]  # Assuming the label is the last character before the extension
                conditions = [4,1,2,4,1,3,2,3,4,1,2,4,1,3,2,3]
                # Get the condition from the file name or directory structure
                condition = conditions[int(file.split('_')[1])]  
                if(label!= '0' and label != '1'):
                    print(file, label)
                f_data = {
                    'roiTimeseries': signal,
                    'pheno': {
                        'subjectId': dir_name,
                        'encoding': None,
                        'label': int(label),
                        'condition': condition,
                        'modality': 'fNIRS'
                    }
                }
                fnirs_data.append(f_data)

    
    #print distribition of labels
    labels = [data['pheno']['label'] for data in fnirs_data]
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print("Label distribution:", label_counts)
    return fnirs_data

def load28(root, type = 'HbR', task = 'nback'):
    fnirs_data = []
    file = os.path.join(root, f'fNIRS28{type}.mat')    
    data = scipy.io.loadmat(file)['simpleData'][0]
    
    for subject in range(len(data)):
      
        time = data[subject][0]
        full_timeseries = data[subject][1]
        #find columns of all 0s
        if subject == 0:
            all_zeros = np.where(np.all(full_timeseries == 0, axis=0))[0]
            print("All zeros: ", all_zeros)
        #remove all 0s
        full_timeseries = np.delete(full_timeseries, all_zeros, axis=1)
        sampling_rate = data[subject][3][0][0]
        zback_stims = data[subject][4][0][0]
        tback_stims = data[subject][4][0][1]
        lhand_stims = data[subject][4][0][2]
        lfoot_stims = data[subject][4][0][3]
        rhand_stims = data[subject][4][0][4]
        rfoot_stims = data[subject][4][0][5]
        tongue_stims = data[subject][4][0][6]
        rest_stims = data[subject][4][0][-1]
        if task == 'nback':
            stims = [zback_stims, tback_stims]
        elif task == 'motor':
            stims = [lhand_stims, rhand_stims]
        
        for num, stims in enumerate(stims):
            for i in range(len(stims[1][0])):
                
                
                start = stims[1][0][i]
                end = start + stims[2][i]
                end = end[0]
                #convert to samples
                # print(sampling_rate, start, end)
                start = int(start * sampling_rate)
                end = int(end * sampling_rate)

                #get the timeseries for the start and end
                timeseries = full_timeseries[start:end]
                if timeseries.shape[0] == 0 or timeseries.shape[0] < 19:
                    # print("Empty timeseries")
                    continue
                if task == 'nback':
                    #get only first 35 samples
                    
                    timeseries = timeseries[:35]
              

                
                #get the subject ID and label from the file name or directory structure
                subject_id = subject
                label = 0 if num == 0 else 1
                encoding = 0 if num < 8 else 1
                condition = i #FOR GNN, not reflective of actual condition
                f_data = {
                        'roiTimeseries': timeseries,
                        'pheno': {
                            'subjectId': subject_id,
                            'encoding': encoding,
                            'label':label,
                            'condition': condition,
                            'modality': 'fNIRS'
                        }}
                fnirs_data.append(f_data)
            
        
        
    all_labels = [data['pheno']['label'] for data in fnirs_data]
    unique, counts = np.unique(all_labels, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))
    return fnirs_data

def getBadChannels(root):
    """
    Get the bad channels (columns with all zeros) from fNIRS HBR data,
    excluding short separation channels (1-based indices).
    """

    file = os.path.join(root, 'fNIRS28HBC.mat')    
    data = scipy.io.loadmat(file)['simpleData'][0]
    
    # Convert short separation channels from 1-based to 0-based indexing
    short_separation_channels_1_based = [22, 26, 36, 55, 65, 83, 102, 108]
    short_separation_channels = [i - 1 for i in short_separation_channels_1_based]
    
    subject = 0
    full_timeseries = data[subject][1]

    # Remove short separation channels
    mask = np.ones(full_timeseries.shape[1], dtype=bool)
    mask[short_separation_channels] = False
    cleaned_timeseries = full_timeseries[:, mask]

    # Find all-zero columns in the cleaned timeseries
    bad_channels = np.where(np.all(cleaned_timeseries == 0, axis=0))[0]

    print("Bad channels (excluding short separation):", bad_channels)

    return bad_channels.tolist()

    

    

 
def calc_MNI_average(digitization):
    """
    Calculate the average of the data in MNI space.
    """
    return np.mean(digitization, axis=0)





def get_parcel_label(mni_coord, atlas_data, affine, radius_mm=30):
    """
    Get the parcel label corresponding to an MNI coordinate.

    Parameters:
    - mni_coord: array-like, shape (3,)
        The MNI coordinate in millimeters.
    - atlas_data: ndarray
        The atlas volume data where each voxel value represents a label.
    - affine: ndarray, shape (4, 4)
        The affine transformation matrix of the atlas.
    - radius_mm: float, optional (default=30)
        The search radius in millimeters for finding the label.

    Returns:
    - label: int or None
        The most common parcel label within the radius, or None if no labels are found.
    """
   
    try:
        # Convert MNI coordinate to voxel indices using the affine matrix
        mni_coord_homogeneous = np.append(mni_coord, 1)
        voxel_coord = np.linalg.inv(affine).dot(mni_coord_homogeneous)[:3]
        voxel_indices = np.round(voxel_coord).astype(int)

        # Calculate radius in voxels
        voxel_sizes = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        radius_voxels = np.ceil(radius_mm / voxel_sizes).astype(int)

        # Create a spherical mask
        ranges = [np.arange(-r, r + 1) for r in radius_voxels]
        grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1)
        distances = np.sqrt(np.sum((grid / voxel_sizes) ** 2, axis=-1))
        mask = distances <= radius_mm

        # Apply mask to calculate sphere coordinates
        sphere_coords = grid[mask] + voxel_indices

        # Filter out-of-bound coordinates
        valid_coords = [
            coord for coord in sphere_coords
            if np.all(coord >= 0) and np.all(coord < atlas_data.shape) and atlas_data[tuple(coord)] != 0
        ]

        if not valid_coords:
            return None

        # Count occurrences of each label within the sphere
        labels, counts = np.unique([atlas_data[tuple(coord)] for coord in valid_coords], return_counts=True)
        

        # 0 = left, 1 = right
        if(mni_coord[0] > 0):
            hemisphere = "right"
        else:
            hemisphere = "left"
        print(mni_coord)
        #print(brodmann_to_name(labels[np.argmax(counts)]), hemisphere)
        return labels[np.argmax(counts)], hemisphere
    except Exception as e:
        print(f"Error: {e}")
        return None

def mni_to_voxel(mni_coords, affine):
    mni_coords = np.array(mni_coords)  # Ensure it's a NumPy array
    voxel_coords = np.linalg.inv(affine).dot(np.append(mni_coords, 1))[:3]
    return np.round(voxel_coords).astype(int)


import nibabel as nib
from scipy.ndimage import distance_transform_edt
import numpy as np

import nibabel as nib
from scipy.spatial.distance import cdist



    



def save_atlas_plot_with_coord(atlas_data, affine, mni_coord, output_path):
    """
    Save a plot of the atlas image with the MNI coordinate overlaid.

    Parameters:
    - atlas_data: ndarray
        The atlas volume data where each voxel value represents a label.
    - affine: ndarray, shape (4, 4)
        The affine transformation matrix of the atlas.
    - mni_coord: array-like, shape (3,)
        The MNI coordinate in millimeters.
    - output_path: str
        Path to save the generated plot.
    """
    # Convert MNI coordinate to voxel indices
    mni_coord_homogeneous = np.append(mni_coord, 1)
    voxel_coord = np.linalg.inv(affine).dot(mni_coord_homogeneous)[:3]
    
    voxel_indices = np.round(voxel_coord).astype(int)
    radius_voxels = int(30 / np.abs(affine[0, 0]))  # Assuming isotropic voxels

    # Create a spherical mask
    x, y, z = np.ogrid[-radius_voxels:radius_voxels+1, -radius_voxels:radius_voxels+1, -radius_voxels:radius_voxels+1]
    mask = x**2 + y**2 + z**2 <= radius_voxels**2

    # Get the coordinates within the sphere
    sphere_coords = np.array(np.where(mask)).T + voxel_indices - radius_voxels

    # Filter out coordinates that are out of bounds and where the atlas value is 0
    valid_coords = [coord for coord in sphere_coords if np.all(coord >= 0) and np.all(coord < atlas_data.shape) and atlas_data[tuple(coord)] != 0]

    # Count the occurrences of each label within the sphere
    labels, counts = np.unique([atlas_data[tuple(coord)] for coord in valid_coords], return_counts=True)

    # Find the label with the most points
    most_common_label = labels[np.argmax(counts)]

    print(f"The most common parcel label within a 30mm radius of {mni_coord} is {most_common_label}.", flush=True)

    # Plot a slice of the atlas and overlay the coordinate
    slice_index = voxel_indices[2]  # Assuming axial slice
   
    print(f"Plot saved to {output_path}")

    import numpy as np

def calculate_average_bold(mni_coord, fmri_data, affine, radius_mm=30):
    """
    Calculate the average BOLD signal within a given radius around an MNI coordinate,
    excluding voxels with all zero values.

    Parameters:
    - mni_coord: array-like, shape (3,)
        The MNI coordinate in millimeters.
    - fmri_data: ndarray
        The fMRI 4D image data where the last dimension is time and earlier dimensions are spatial.
    - affine: ndarray, shape (4, 4)
        The affine transformation matrix of the fMRI image.
    - radius_mm: float, optional (default=30)
        The radius in millimeters for averaging.

    Returns:
    - average_bold: ndarray, shape (n_timepoints,)
        The average BOLD signal within the radius across all timepoints.
    """
 

    # Step 1: Convert MNI coordinate to voxel indices
    mni_coord_homogeneous = np.append(mni_coord, 1)  # Add homogeneous coordinate
    voxel_coord = np.linalg.inv(affine).dot(mni_coord_homogeneous)[:3]
    voxel_indices = np.round(voxel_coord).astype(int)

    # Step 2: Calculate the radius in voxel units
    voxel_sizes = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))  # Voxel dimensions
    radius_voxels = np.ceil(radius_mm / voxel_sizes).astype(int)

    # Step 3: Generate a spherical mask
    ranges = [np.arange(-r, r + 1) for r in radius_voxels]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1)  # Create a grid of offsets
    distances = np.sqrt(np.sum((grid * voxel_sizes) ** 2, axis=-1))  # Calculate distances in mm
    mask = distances <= radius_mm

    # Step 4: Extract voxel coordinates within the sphere
    sphere_coords = grid[mask] + voxel_indices  # Add sphere offsets to the center voxel
    sphere_coords = sphere_coords[
        (sphere_coords >= 0).all(axis=1) & (sphere_coords < fmri_data.shape[:3]).all(axis=1)
    ]  # Bounds checking

    if len(sphere_coords) == 0:
        raise ValueError("No valid voxels found within the specified radius. Check the inputs.")

    # Step 5: Extract signal values at valid voxel coordinates
    bold_signals = fmri_data[sphere_coords[:, 0], sphere_coords[:, 1], sphere_coords[:, 2], :]

    # Step 6: Exclude voxels with all zero values across time
    non_zero_mask = np.any(bold_signals != 0, axis=1)
    valid_bold_signals = bold_signals[non_zero_mask]

    if valid_bold_signals.size == 0:
        raise ValueError("All voxels within the specified radius have zero values.")

    # Step 7: Compute the average BOLD signal across valid voxels
    average_bold = np.mean(valid_bold_signals, axis=0)

    return average_bold



def parcel_num_to_name(num):
    code_to_region = {
    2001: "Precentral_L", 2002: "Precentral_R", 2101: "Frontal_Sup_L", 2102: "Frontal_Sup_R",
    2111: "Frontal_Sup_Orb_L", 2112: "Frontal_Sup_Orb_R", 2201: "Frontal_Mid_L", 2202: "Frontal_Mid_R",
    2211: "Frontal_Mid_Orb_L", 2212: "Frontal_Mid_Orb_R", 2301: "Frontal_Inf_Oper_L", 2302: "Frontal_Inf_Oper_R",
    2311: "Frontal_Inf_Tri_L", 2312: "Frontal_Inf_Tri_R", 2321: "Frontal_Inf_Orb_L", 2322: "Frontal_Inf_Orb_R",
    2331: "Rolandic_Oper_L", 2332: "Rolandic_Oper_R", 2401: "Supp_Motor_Area_L", 2402: "Supp_Motor_Area_R",
    2501: "Olfactory_L", 2502: "Olfactory_R", 2601: "Frontal_Sup_Medial_L", 2602: "Frontal_Sup_Medial_R",
    2611: "Frontal_Med_Orb_L", 2612: "Frontal_Med_Orb_R", 2701: "Rectus_L", 2702: "Rectus_R",
    3001: "Insula_L", 3002: "Insula_R", 4001: "Cingulum_Ant_L", 4002: "Cingulum_Ant_R",
    4011: "Cingulum_Mid_L", 4012: "Cingulum_Mid_R", 4021: "Cingulum_Post_L", 4022: "Cingulum_Post_R",
    4101: "Hippocampus_L", 4102: "Hippocampus_R", 4111: "ParaHippocampal_L", 4112: "ParaHippocampal_R",
    4201: "Amygdala_L", 4202: "Amygdala_R", 5001: "Calcarine_L", 5002: "Calcarine_R",
    5011: "Cuneus_L", 5012: "Cuneus_R", 5021: "Lingual_L", 5022: "Lingual_R",
    5101: "Occipital_Sup_L", 5102: "Occipital_Sup_R", 5201: "Occipital_Mid_L", 5202: "Occipital_Mid_R",
    5301: "Occipital_Inf_L", 5302: "Occipital_Inf_R", 5401: "Fusiform_L", 5402: "Fusiform_R",
    6001: "Postcentral_L", 6002: "Postcentral_R", 6101: "Parietal_Sup_L", 6102: "Parietal_Sup_R",
    6201: "Parietal_Inf_L", 6202: "Parietal_Inf_R", 6211: "SupraMarginal_L", 6212: "SupraMarginal_R",
    6221: "Angular_L", 6222: "Angular_R", 6301: "Precuneus_L", 6302: "Precuneus_R",
    6401: "Paracentral_Lobule_L", 6402: "Paracentral_Lobule_R", 7001: "Caudate_L", 7002: "Caudate_R",
    7011: "Putamen_L", 7012: "Putamen_R", 7021: "Pallidum_L", 7022: "Pallidum_R",
    7101: "Thalamus_L", 7102: "Thalamus_R", 8101: "Heschl_L", 8102: "Heschl_R",
    8111: "Temporal_Sup_L", 8112: "Temporal_Sup_R", 8121: "Temporal_Pole_Sup_L", 8122: "Temporal_Pole_Sup_R",
    8201: "Temporal_Mid_L", 8202: "Temporal_Mid_R", 8211: "Temporal_Pole_Mid_L", 8212: "Temporal_Pole_Mid_R",
    8301: "Temporal_Inf_L", 8302: "Temporal_Inf_R", 9001: "Cerebelum_Crus1_L", 9002: "Cerebelum_Crus1_R",
    9011: "Cerebelum_Crus2_L", 9012: "Cerebelum_Crus2_R", 9021: "Cerebelum_3_L", 9022: "Cerebelum_3_R",
    9031: "Cerebelum_4_5_L", 9032: "Cerebelum_4_5_R", 9041: "Cerebelum_6_L", 9042: "Cerebelum_6_R",
    9051: "Cerebelum_7b_L", 9052: "Cerebelum_7b_R", 9061: "Cerebelum_8_L", 9062: "Cerebelum_8_R",
    9071: "Cerebelum_9_L", 9072: "Cerebelum_9_R", 9081: "Cerebelum_10_L", 9082: "Cerebelum_10_R",
    9100: "Vermis_1_2", 9110: "Vermis_3", 9120: "Vermis_4_5", 9130: "Vermis_6",
    9140: "Vermis_7", 9150: "Vermis_8", 9160: "Vermis_9", 9170: "Vermis_10"
    }
    return code_to_region[num]

def brodmann_to_name(num):
    brodmann_areas = {
    1: "Primary somatosensory cortex",
    2: "Primary somatosensory cortex",
    3: "Primary somatosensory cortex",
    4: "Primary motor cortex",
    5: "Somatosensory association cortex",
    6: "Premotor cortex and supplementary motor cortex",
    7: "Somatosensory association cortex",
    8: "Includes frontal eye fields",
    9: "Dorsolateral prefrontal cortex",
    10: "Anterior prefrontal cortex",
    11: "Orbitofrontal area",
    12: "Orbitofrontal area",
    13: "Insular cortex",
    14: "Insular cortex",
    15: "Anterior temporal lobe",
    16: "Insular cortex",
    17: "Primary visual cortex",
    18: "Secondary visual cortex",
    19: "Associative visual cortex",
    20: "Inferior temporal gyrus",
    21: "Middle temporal gyrus",
    22: "Superior temporal gyrus",
    23: "Ventral posterior cingulate cortex",
    24: "Ventral anterior cingulate cortex",
    25: "Subgenual area",
    26: "Ectosplenial portion of the retrosplenial region of the cerebral cortex",
    27: "Piriform cortex",
    28: "Ventral entorhinal cortex",
    29: "Retrosplenial cingulate cortex",
    30: "Part of cingulate cortex",
    31: "Dorsal posterior cingulate cortex",
    32: "Dorsal anterior cingulate cortex",
    33: "Part of anterior cingulate cortex",
    34: "Dorsal entorhinal cortex",
    35: "Perirhinal cortex",
    36: "Ectorhinal area",
    37: "Fusiform gyrus",
    38: "Temporopolar area",
    39: "Angular gyrus",
    40: "Supramarginal gyrus",
    41: "Auditory cortex",
    42: "Auditory cortex",
    43: "Primary gustatory cortex",
    44: "Pars opercularis",
    45: "Pars triangularis",
    46: "Dorsolateral prefrontal cortex",
    47: "Pars orbitalis, part of the inferior frontal gyrus",
    48: "Retrosubicular area",
    49: "Parasubicular area (in rodents)",
    52: "Parainsular area"
    }
    return brodmann_areas[num]


import torch
#test load 
if __name__ == '__main__':
    # #test load
    # data_dir = "Dataset/Data/fNIRS/Preprocessed/"
    # data = load(data_dir, type='HbR')

    # groups = [x['pheno']['subjectId'] for x in data]
    # arrs = [x['roiTimeseries'] for x in data]
    # #print the unque groups
   

    # #check the shape of x[0]
    # print(arrs[0].shape)
    data_dir = "./Dataset/Data/fNIRS/fNIRS28-1.38/"
    data = load28(data_dir, type='HbC')
    print(data[:5])
    torch.save(data, "Dataset/Data/fNIRS/fNIRS28-1.38.save")
    np.save("Dataset/Data/fNIRS/fNIRS28-1.38.npy", data)
    torch.load("Dataset/Data/fNIRS/fNIRS28-1.38.save")
    
    