
import os
import numpy as np
from nilearn import datasets, image, plotting
from nilearn.regions import connected_regions
from scipy.ndimage import center_of_mass
from random import sample
import matplotlib.pyplot as plt
datadir = "./Dataset/Data"



import os
import numpy as np
from nilearn import datasets, image
from .fnirs_utils import get_parcel_label, save_atlas_plot_with_coord
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
    print(atlas_data.max())
    for i in range(1, int(atlas_data.max()) + 1):  # Iterate over all region indices
        region_mask = atlas_data == i  # Create a binary mask for the current region
        if np.any(region_mask):  # Skip empty regions
            com_voxel = center_of_mass(region_mask)
            com_mni = image.coord_transform(
                com_voxel[0], com_voxel[1], com_voxel[2], atlas_img.affine
            )
            mni_coords.append(com_mni)

    return mni_coords
    
def prep_atlas(atlas, datadir, mni_coords=None):
    """
    Load or download the specified atlas and optionally filter ROIs based on MNI coordinates.

    Parameters:
    - atlas (str): Name of the atlas to load ("schaefer7_400" or "AAL").
    - datadir (str): Directory where atlases will be stored.
    - mni_coords (list of list): List of MNI coordinates to filter ROIs. Default is None.

    Returns:
    - filtered_atlas_img (Nifti1Image): Filtered atlas image containing only ROIs covering the MNI coordinates.
    """
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

    else:
        raise ValueError(f"Atlas '{atlas}' is not recognized. Choose 'schaefer7_400' or 'AAL'.")

    if mni_coords is None:
        # No filtering, return the full atlas
        # return atlas_img
        return atlas_img
    
    # Convert MNI coordinates to voxel space and filter ROIs
    atlas_data = atlas_img.get_fdata()
    roi_indices = []

    for coord in mni_coords:
        label = get_parcel_label(coord, atlas_data, atlas_img.affine)
        roi_indices.append(label)

    # Unique ROI indices
    roi_indices = np.unique(roi_indices)

    # Create a filtered atlas
    filtered_atlas_data = np.isin(atlas_data, roi_indices) * atlas_data
    filtered_atlas_img = image.new_img_like(atlas_img, filtered_atlas_data)

    return filtered_atlas_img



    