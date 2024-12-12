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



import numpy as np

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
        print(f"The most common parcel label within a 30mm radius of {mni_coord} is {labels[np.argmax(counts)]}.", flush=True)
        return labels[np.argmax(counts)]
    except Exception as e:
        print(f"Error: {e}")
        return None


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
    Calculate the average BOLD signal within a given radius around an MNI coordinate.

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
    valid_coords = [
        coord for coord in sphere_coords
        if np.all(coord >= 0) and np.all(coord < fmri_data.shape[:3])  # Bounds checking
    ]

    if not valid_coords:
        raise ValueError("No valid voxels found within the specified radius.")

    # Step 5: Extract signal values at valid voxel coordinates
    bold_signals = np.array([fmri_data[tuple(coord)] for coord in valid_coords])

    # Step 6: Compute the average BOLD signal across valid voxels
    average_bold = np.mean(bold_signals, axis=0)

    return average_bold
