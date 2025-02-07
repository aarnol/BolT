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
    image_path = os.path.join(folder_path, "tfMRI_MOTOR_RL.nii.gz")  # Adjusted for motor task

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
        all_labels = []

        stimulus_types = ['lh','rh']
        conditions = ['left', 'right']

        # Process stimuli and conditions
        for stimulus in stimulus_types:
          
            path = os.path.join(folder_path, "metadata", f"{stimulus}.txt")

            if not os.path.exists(path):
                print(f"Missing files for {stimulus} in {folder_path}. Skipping...")
                continue

            for i in range(2):
                
                try:
                    data = np.loadtxt(path)
                    
                    all_onsets.append(float(data[i][0]))  # Append onsets
                    all_durations.append(float(data[i][1]))  # Append durations
                    all_labels.append("left" if stimulus == "lh" else "right")

                except Exception as e:
                    print(f"Error loading data from {path}: {e}")
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
        print("Design matrix (left and right columns):")
        print(design_matrix[['left', 'right']].head(20))


        print("Design matrix created")
        print("Design matrix columns:", design_matrix.columns)

        print(f"Number of scans: {n_scans}")
        print(f"Design matrix shape: {design_matrix.shape}")
        print(f"Image shape: {image.shape}")

        if design_matrix.isnull().values.any():
            print("Warning: NaNs detected in the design matrix. Replacing with zeros.")
            design_matrix = design_matrix.fillna(0)
        

        # Fit the first-level GLM
        glm = FirstLevelModel(t_r=t_r)
        print("glm created")
        glm = glm.fit(image, design_matrices=design_matrix)
        print("GLM fitted")
        

      

        contrast = glm.compute_contrast("left-right", output_type='z_score')

        # Define and compute contrast
        # contrast = glm.compute_contrast('left - right', output_type='z_score')
        print("Contrast computed")

        # Save the contrast image
        contrast_image_path = os.path.join(folder_path, "contrast_left_vs_right.nii.gz")
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
        p_val = 0.0001
        p001_unc = norm.isf(p_val)
        z_map_path = os.path.join(root, "second_level_motor.png")
        plot_glass_brain(z_map, threshold=p001_unc, display_mode="ortho", 
                         colorbar=True, output_file=z_map_path, plot_abs=False)
        print(f"Second-level result saved to {z_map_path}")
    except Exception as e:
        print(f"Error during second-level analysis: {e}")
else:
    print("No contrasts computed. Second-level analysis skipped.")
