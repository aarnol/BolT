import os
import numpy as np
import nibabel as nib
import concurrent.futures
import torch
import pandas as pd
import boto3
import math

# Constants
root_dir = "/scratch/alpine/alar6830/motor_labeled/"
local_root = "/scratch/alpine/alar6830/motor/"
encodings = ["LR", "RL"]
bucket_name = 'hcp-openaccess'
behavior_data = pd.read_csv("unrestricted_hcp_freesurfer.csv")
subject_values = behavior_data['Subject'].values

# Initialize S3 client
s3_client = boto3.client('s3')

def frame_to_seconds(frame):
    return math.ceil(frame * 0.72)

def seconds_to_frame(sec):
    return math.ceil(sec / 0.72)

def check_s3_path_exists(bucket_name, path):
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=path)
        return 'Contents' in response
    except Exception as e:
        print(f"Error checking path {path}: {e}")
        return False

def download_file_from_s3(bucket_name, s3_key, local_path):
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        print(f"Downloaded {s3_key} to {local_path}")
    except Exception as e:
        print(f"Error downloading file {s3_key}: {e}")

def process_subject(subject, enc):
    try:
        subject = str(subject)
        for trial_idx in range(2):
            if os.path.exists(os.path.join(root_dir, f"{subject}{enc}{trial_idx}.nii.gz")):
                return  # Skip if already exists

        enc_num = 0 if enc == "LR" else 1
        s3_key = f"HCP_1200/{subject}/MNINonLinear/Results/tfMRI_MOTOR_{enc}/tfMRI_MOTOR_{enc}.nii.gz"
        local_path = os.path.join(local_root, subject, enc, f"tfMRI_MOTOR_{enc}.nii.gz")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if not os.path.exists(local_path):
            if check_s3_path_exists(bucket_name, s3_key):
                download_file_from_s3(bucket_name, s3_key, local_path)
            else:
                print(f"S3 path does not exist: {s3_key}")
                return

        img = nib.load(local_path)
        full_data = img.get_fdata()

        # Load onset files
        onset_dir = os.path.join(local_root, subject, enc, "metadata")
        os.makedirs(onset_dir, exist_ok=True)
        ev_s3_path = f"HCP_1200/{subject}/MNINonLinear/Results/tfMRI_MOTOR_{enc}/EVs/"

        # Download EVs if not present
        for label in ["t", "rh", "rf", "lh", "lf"]:
            ev_file = os.path.join(onset_dir, f"{label}.txt")
            if not os.path.exists(ev_file):
                try:
                    s3_client.download_file(bucket_name, f"{ev_s3_path}{label}.txt", ev_file)
                except Exception as e:
                    print(f"Failed to download EV file {label}.txt: {e}")
                    return

        # Process onset files
        labels = [np.loadtxt(os.path.join(onset_dir, f"{l}.txt")) for l in ["t", "rh", "rf", "lh", "lf"]]

        for i, value in enumerate(labels):
            try:
                for j in range(2):
                    onset = seconds_to_frame(value[j][0])
                    offset = seconds_to_frame(value[j][0] + value[j][1])

                    cropped_image = full_data[:, :, :, onset:offset]
                    cropped_img = nib.Nifti1Image(cropped_image, img.affine)

                    output_path = os.path.join(root_dir, f"{subject}{enc_num}{j}{i}.nii.gz")
                    nib.save(cropped_img, output_path)
            except Exception as e:
                print(f"Problem with onsets for {subject} {enc}: {e}")

    except Exception as e:
        print(f"Error processing subject {subject} {enc}: {e}")

# Run with threading
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_subject, subject, enc) for subject in subject_values for enc in encodings]
    for future in concurrent.futures.as_completed(futures):
        future.result()
