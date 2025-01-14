import os
import numpy as np
import nibabel as nib
import concurrent.futures
import torch
import pandas as pd


# Define the root directory
root_dir = "/scratch/alpine/alar6830/motor_labeled/"
meta_dir = "/scratch/alpine/alar6830/motor"
encodings = ["LR", "RL"]

# Function to process each subject and encoding
def process_subject(subject, enc):
    try:
        if os.path.exists(os.path.join(root_dir, f"{subject}141.nii.gz")):
            print(f"done with {subject}", flush =True)
            return

        image_path = f"/scratch/alpine/alar6830/motor/{subject}/{enc}/tfMRI_MOTOR_{enc}.nii.gz"
        img = nib.load(image_path)
        full_data = img.get_fdata()  # Convert to NumPy array
        


        # Path to onset files
        onset_paths = f"/scratch/alpine/alar6830/motor/{subject}/{enc}/metadata/"
        if not os.path.exists(onset_paths):
 
            os.mkdir(onset_paths)
            aws_command = f"aws s3 cp s3://hcp-openaccess/HCP_1200/{subject}/MNINonLinear/Results/tfMRI_MOTOR_{enc}/EVs/ {onset_paths} --recursive"
            os.system(aws_command)

        # Load onset files
        tongue = np.loadtxt(onset_paths + "t.txt")
        rh = np.loadtxt(onset_paths + "rh.txt")
        rf = np.loadtxt(onset_paths + "rf.txt")
        lh = np.loadtxt(onset_paths + "lh.txt")
        lf = np.loadtxt(onset_paths + "lf.txt")

        data = [tongue,rh,rf,lh,lf]
        print(data, flush = True)
        i = 0
        for value in data:
            try:
                for j in range(2):
                    print(value, flush = True)
                    onset = seconds_to_frame(value[j][0])  
                    offset = seconds_to_frame(value[j][1] + value[j][0])

                    cropped_image = full_data[:, :, :, onset:offset]

                    img = nib.Nifti1Image(cropped_image, img.affine)

                    # Determine encoding number (0 for LR, 1 for RL)
                    enc_num = 0 if enc == "LR" else 1

                    # Save the cropped image
                    nib.save(img, os.path.join(root_dir, f"{subject}{enc_num}{i}.nii.gz"))
                i+=1
            except Exception as e:
                print(f"Onsets {value} not properly formatted for {subject}: {e}")
    except Exception as e:
        print(f"Error processing {subject} with encoding {enc}: {e}", flush = True)

import math
## if onset time is between frames, use the next available frame

def frame_to_seconds(frame):
    return math.ceil * 0.72

def seconds_to_frame(sec):
    return math.ceil(sec/.72)

behavior_data = pd.read_csv("unrestricted_hcp_freesurfer.csv")
subject_values = behavior_data['Subject'].values

# Use ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    print("starting threads", flush = True)
    # Submit tasks for each subject and encoding combination
    futures = []
    for subject in subject_values:
        for enc in encodings:
            futures.append(executor.submit(process_subject, subject, enc))
         

    # Wait for all tasks to complete
    for future in concurrent.futures.as_completed(futures):
        future.result()  # This will raise exceptions if any occurred in the thread