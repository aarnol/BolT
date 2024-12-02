import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.datasets import fetch_atlas_aal


def get_parcel_label(mni_coord, atlas_data, affine):
    # Convert MNI coordinate to voxel indices
    voxel_indices = np.linalg.inv(affine).dot(np.append(mni_coord, 1))[:3]
    print(voxel_indices)
    voxel_indices = np.round(voxel_indices).astype(int)
    
    # Extract the label at the voxel indices
    label = atlas_data[tuple(voxel_indices)]
    return label

if __name__ == "__main__":
    # Load the Schaefer atlas
    atlas = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    atlas_filename = atlas['maps']

    # Load the atlas NIfTI file
    atlas_img = nib.load(atlas_filename)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    # Load the AAL atlas
    aal_atlas = fetch_atlas_aal()
    aal_atlas_filename = aal_atlas['maps']
    print(aal_atlas.keys()) 

    # Load the AAL atlas NIfTI file
    aal_atlas_img = nib.load(aal_atlas_filename)
    aal_atlas_data = aal_atlas_img.get_fdata()
    aal_affine = aal_atlas_img.affine
    # Function to find the label of an MNI coordinate
    # Example MNI coordinate
    mni_coord = [30, -50, 40]  # Replace with your coordinate
    label = get_parcel_label(mni_coord, atlas_data, affine)

    print(f"The MNI coordinate {mni_coord} falls into parcel label {int(label)}.")
