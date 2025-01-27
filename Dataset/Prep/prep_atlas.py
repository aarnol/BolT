
import os
import numpy as np
from nilearn import datasets, image, plotting
from nilearn.regions import connected_regions
from scipy.ndimage import center_of_mass
from random import sample
import matplotlib.pyplot as plt
import nibabel as nib
datadir = "./Dataset/Data"



import os
import numpy as np
from nilearn import datasets, image
from fnirs_utils import get_parcel_label, save_atlas_plot_with_coord
def calc_COM(atlas_img):
    """
    Calculate the center of mass (COM) for each region in the given atlas image.

    Parameters:
    - atlas_img (Nifti1Image): The atlas image.

    Returns:
    - mni_coords (list of tuple): List of MNI coordinates for the center of mass of each region.
    """
    atlas_data = atlas_img.get_fdata()
    mni_coords = []
    
    for i in range(1, int(atlas_data.max()) + 1):  # Iterate over all region indices
        region_mask = atlas_data == i  # Create a binary mask for the current region
        if np.any(region_mask):  # Skip empty regions
            com_voxel = center_of_mass(region_mask)
            com_mni = image.coord_transform(
                com_voxel[0], com_voxel[1], com_voxel[2], atlas_img.affine
            )
            mni_coords.append(com_mni)

    return mni_coords
    
def prep_atlas(atlas, datadir = datadir, mni_coords=None):
    """
    Load or download the specified atlas and optionally filter ROIs based on MNI coordinates.

    Parameters:
    - atlas (str): Name of the atlas to load ("schaefer7_400" or "AAL").
    - datadir (str): Directory where atlases will be stored.
    - mni_coords (list of list): List of MNI coordinates to filter ROIs. Default is None.

    Returns:
    - filtered_atlas_img (Nifti1Image): Filtered atlas image containing only ROIs covering the MNI coordinates.
    """
    if(atlas == "sphere"):
        return None
    atlas_path = os.path.join(datadir, "Atlasses", atlas)
    os.makedirs(atlas_path, exist_ok=True)  # Ensure the directory exists

    # Load the atlas
    if atlas == "schaefer7_400":
        atlasInfo = datasets.fetch_atlas_schaefer_2018(
            n_rois=400, 
            yeo_networks=7, 
            resolution_mm=2, 
            data_dir=datadir + "/Atlasses"
        )
        atlas_img = image.load_img(atlasInfo["maps"])

    elif atlas == "AAL":
        atlasInfo = datasets.fetch_atlas_aal(data_dir=datadir + "/Atlasses")
        atlas_img = image.load_img(atlasInfo["maps"])
    elif atlas == "brodmann":
        atlas_img =  nib.load(os.path.join("Dataset", "Data", "Atlasses", "brodmann.nii.gz"))
    else:
        raise ValueError(f"Atlas '{atlas}' is not recognized. Choose 'schaefer7_400' or 'AAL'.")
    
    return atlas_img



    