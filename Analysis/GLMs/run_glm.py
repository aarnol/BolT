import nilearn
import numpy as np
import nibabel as nib
import argparse
import os
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_glass_brain
from scipy.stats import norm

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", required=True, help="Directory containing subject data")
parser.add_argument("-u", "--user", required=True, help="User's scratch directory")
parser.add_argument("-s", "--smoothing", type=float, required=True, help="Smoothing kernel size in mm")

# Parse arguments
args = parser.parse_args()
root = os.path.join("/scratch/alpine/", args.user, args.directory)

# Define repetition time and initialize variables
t_r = 0.72
contrasts = []

# Process each folder in the directory
for folder in os.listdir(root):
    folder_path = os.path.join(root, folder)
    image_path = os.path.join(folder_path, "tfMRI_WM_RL.nii.gz")

    if not os.path.exists(image_path):
        print(f"Functional image not found in {folder_path}. Skipping...")
        continue

    try:
        # Load and smooth the functional image
        image = nib.load(image_path)
        print(f"Image loaded from {image_path}")
        image = nilearn.image.smooth_img(image, fwhm=args.smoothing)
        print(f"Image smoothed with FWHM = {args.smoothing} mm")

        # Initialize lists to store onsets and durations
        all_onsets = []
        all_durations = []
        all_labels = []  # To track condition labels

        stimulus_types = ['faces', 'places', 'tools', 'body']
        conditions = ['0bk', '2bk']

        # Process stimuli and conditions
        for stimulus in stimulus_types:
            

            paths = {cond: os.path.join(folder_path, f"{cond}_{stimulus}.txt") for cond in conditions}

            if not all(os.path.exists(path) for path in paths.values()):
                print(f"Missing files for {stimulus} in {folder_path}. Skipping...")
                continue

            for cond in conditions:
                try:
                    data = np.loadtxt(paths[cond])
                    print(data, flush = True)
                    all_onsets.append(data[0])  # Append onsets
                    all_durations.append(data[1])  # Append durations
                    if cond == "0bk":
                        all_labels.append("zero")
                    else:
                        all_labels.append("two")
                    
                except Exception as e:
                    print(f"Error loading data from {paths[cond]}: {e}")
                    continue

        # Create events dataframe
        events = pd.DataFrame({
            "trial_type": all_labels,
            "onset": all_onsets,
            "duration": all_durations
        })
        print(events)

        # Define frame times
        n_scans = image.shape[-1]
        frame_times = np.arange(n_scans) * t_r

        # Create design matrix
        design_matrix = make_first_level_design_matrix(
            frame_times, events, drift_model='polynomial', drift_order=3
        )
        print("Design matrix created")

        # Fit the first-level GLM
        glm = FirstLevelModel(t_r=t_r)
        glm = glm.fit(image, design_matrices=design_matrix)
        print("GLM fitted")

        # Define and compute contrast
        contrast = glm.compute_contrast('two - zero', output_type='z_score')
        print("Contrast computed")

        # Save the contrast image
        contrast_image_path = os.path.join(folder_path, "contrast_two_vs_zero.nii.gz")
        contrast.to_filename(contrast_image_path)
        print(f"Contrast image saved to {contrast_image_path}")

        contrasts.append(contrast)

    except Exception as e:
        print(f"Error processing {folder_path}: {e}")

# Perform second-level analysis
if contrasts:
    try:
        design_matrix = pd.DataFrame([1] * len(contrasts), columns=["intercept"])
        second_level_model = SecondLevelModel()
        second_level_model.fit(contrasts, design_matrix=design_matrix)
        
        z_map = second_level_model.compute_contrast(output_type="z_score")

        # Threshold and save the second-level result
        p_val = 0.001
        p001_unc = norm.isf(p_val)
        z_map_path = os.path.join(root, "second_level.png")
        plot_glass_brain(z_map, threshold=p001_unc, display_mode="ortho", 
                         colorbar=True, output_file=z_map_path, plot_abs=False)
        print(f"Second-level result saved to {z_map_path}")
    except Exception as e:
        print(f"Error during second-level analysis: {e}")
else:
    print("No contrasts computed. Second-level analysis skipped.")
