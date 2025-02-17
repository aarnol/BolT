import sys

import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils
from matplotlib import pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img
data_path = os.path.join(os.path.dirname(__file__), '10Examples', "100206", "tfMRI_WM_RL.nii.gz")

# Load the data

fnirs_data, MNI = fnirs_utils.load_fnirs(os.path.join(os.path.dirname(__file__), 'fNIRS'))
import nibabel as nib
from nilearn import datasets
from nilearn.plotting import plot_glass_brain
import numpy as np
image =  nib.load(data_path)
image = mean_img(image)
affine = image.affine
data = image.get_fdata()



def get_closest_voxel_distance(mni_coords, affine, image_data):
    """ Calculate the distance from MNI coordinates to the closest voxel in millimeters. """
    voxel_coords = fnirs_utils.mni_to_voxel(mni_coords, affine)
    voxel_coords = np.round(voxel_coords).astype(int)
    
    # Ensure voxel coordinates are within the image bounds
    voxel_coords = np.clip(voxel_coords, 0, np.array(image_data.shape) - 1)
    
    # Convert voxel coordinates back to MNI space to get the exact position
    closest_voxel_mni = nib.affines.apply_affine(affine, voxel_coords)
    
    # Calculate the Euclidean distance in millimeters
    distance = np.linalg.norm(np.array(mni_coords) - np.array(closest_voxel_mni))
    
    return distance
for coord in MNI:
    print("Distance from MNI to closest voxel (mm):", get_closest_voxel_distance(coord, affine, data))
# Example usage
mni = MNI[0]
distance = get_closest_voxel_distance(mni, affine)
print("Distance from MNI to closest voxel (mm):", distance)



# # Get the first mni coordinate

# mni = MNI[0]

# # project the mni coordinate to the image

# # coord = fnirs_utils.project_mni_to_surface_nifti(mni, data, affine)
# # Example usage
# brain_mask_path = data_path  # Provide NIfTI brain mask
# import mne

# subjects_dir = mne.datasets.fetch_fsaverage()
# print("Subjects Directory:", subjects_dir)



# projected_coord = fnirs_utils.project_mni_to_surface_com(mni, brain_mask_path)
# print("Projected MNI Coordinate:", projected_coord)

# #plot the image with the orignal mni coordinate and the projected mni coordinate
# def plot_mni_on_nifti(nifti_path, mni_coords_list):
#     """ Plot MNI coordinates on a NIfTI image. """
#     # Load NIfTI image
#     img = nib.load(nifti_path)
    
#     # Convert MNI to voxel coordinates
#     voxel_coords_list = [fnirs_utils.mni_to_voxel(mni, img.affine) for mni in mni_coords_list]
    
#     # Plot the image
#     display = plot_glass_brain(mean_img(img), title="MNI Coordinates on NIfTI", display_mode="lyr")
    
#     # Overlay points
#     display.add_markers([mni_coords_list[0]], marker_color='red', marker_size=100)
#     display.add_markers([mni_coords_list[1]], marker_color='blue', marker_size=100)
    
#     plt.show()
# def plot_mni_on_example_brain(mni_coords_list):
#     """ Plot MNI coordinates on an example brain. """
    
#     # Load an example brain image
#     example_brain = datasets.load_mni152_template()
    
#     # Plot the example brain
#     display = plot_glass_brain(example_brain, title="MNI Coordinates on Example Brain", display_mode='x')
    
#     # Overlay points
    
#     display.add_markers([mni_coords_list[0]], marker_color='red', marker_size=100)
#     display.add_markers([mni_coords_list[1]], marker_color='blue', marker_size=100)
    
#     plt.show()

#     # Example usage
# plot_mni_on_example_brain([mni, projected_coord])
# plot_mni_on_nifti(data_path, [mni, projected_coord])
