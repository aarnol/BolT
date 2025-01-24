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
    folder_path = os.path.join(root, folder, "RL")
    image_path = os.path.join(folder_path, "tfMRI_MOTOR_RL.nii.gz")  # Adjusted for Motor task

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

        # Define Motor task conditions
        # "other conditions 'rf', 'lh', 'rh', 't"
        motor_conditions = ['rh', 'cue']

        # Process conditions
        for condition in motor_conditions:
            file_path = os.path.join(folder_path, "metadata", f"{condition}.txt")

            if not os.path.exists(file_path):
                print(f"Missing file for {condition} in {folder_path}. Skipping...")
                continue

            try:
                data = np.loadtxt(file_path)
                if(condition == 'cue'):
                    all_onsets.append(0.0)
                    all_durations.append(data[0][0] + data[0][1])
                    all_labels.append("rest")
                    continue
                print(data, flush=True)
                all_onsets.append(data[0][0])  # Append onsets
                all_durations.append(data[0][1])  # Append durations
                all_labels.append(condition)  # Append condition label
            except Exception as e:
                print(f"Error loading data from {file_path}: {e}")
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

        # Define and compute contrasts
        contrast = glm.compute_contrast('rh - rest', output_type='z_score')  # Example contrast
        print("Contrast computed")

        # Save the contrast image
        contrast_image_path = os.path.join(folder_path, "contrast_left_vs_rest.nii.gz")
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
