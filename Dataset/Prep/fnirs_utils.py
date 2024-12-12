import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
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
        print(blocks)
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
    """
    Get the parcel label corresponding to an MNI coordinate.

    Parameters:
    - mni_coord: array-like, shape (3,)
        The MNI coordinate in millimeters.
    - atlas_data: ndarray
        The atlas volume data where each voxel value represents a label.
    - affine: ndarray, shape (4, 4)
        The affine transformation matrix of the atlas.

    Returns:
    - label: int
        The parcel label at the given MNI coordinate.
    """
    # Convert MNI coordinate to voxel indices using the affine matrix
    mni_coord_homogeneous = np.append(mni_coord, 1)  # Make it homogeneous by adding a 1
    voxel_coord = np.linalg.inv(affine).dot(mni_coord_homogeneous)[:3]
    voxel_indices = np.round(voxel_coord).astype(int)

    # Extract the label at the voxel indices
    try:
        label = atlas_data[tuple(voxel_indices)]
        print(f"The MNI coordinate {mni_coord} falls into parcel label {int(label)}.", flush=True)
        if(label == 0):
            save_atlas_plot_with_coord(atlas_data, affine, mni_coord, "mni.png")
    except IndexError:
        raise ValueError(f"The MNI coordinate {mni_coord} is outside the bounds of the atlas data.")

    return label

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

    # Plot a slice of the atlas and overlay the coordinate
    slice_index = voxel_indices[2]  # Assuming axial slice
    plt.figure(figsize=(8, 8))
    plt.imshow(atlas_data[:, :, slice_index], cmap="gray", origin="lower")
    plt.scatter(voxel_indices[0], voxel_indices[1], color="red", label="MNI Coordinate")
    plt.title(f"Atlas Slice with MNI Coordinate {mni_coord}")
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")