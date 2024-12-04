
import os 
import nilearn as nil
import nilearn.datasets


datadir = "./Dataset/Data"



import os
import numpy as np
from nilearn import datasets, image

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
        return atlas_img

    # Convert MNI coordinates to voxel space and filter ROIs
    atlas_data = atlas_img.get_fdata()
    roi_indices = []

    for coord in mni_coords:
        # Convert MNI to voxel coordinates
        voxel_coord = np.linalg.inv(atlas_img.affine).dot(np.append(coord, 1))[:3]
        voxel_coord = np.round(voxel_coord).astype(int)

        # Get the ROI index at the voxel
        try:
            roi_index = atlas_data[tuple(voxel_coord)]
            if roi_index > 0:  # Exclude background
                roi_indices.append(int(roi_index))
        except IndexError:
            print(f"Warning: Coordinate {coord} is out of bounds for the atlas.")
            continue

    # Unique ROI indices
    roi_indices = np.unique(roi_indices)

    # Create a filtered atlas
    filtered_atlas_data = np.isin(atlas_data, roi_indices) * atlas_data
    filtered_atlas_img = image.new_img_like(atlas_img, filtered_atlas_data)

    return filtered_atlas_img



    