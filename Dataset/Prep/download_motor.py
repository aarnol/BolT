import os


import pandas as pd

import os
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Load behavior data
behavior_data = pd.read_csv("unrestricted_hcp_freesurfer.csv")
subject_values = behavior_data['Subject'].values
print(f"Total subjects: {len(subject_values)}")

bucket_name = 'hcp-openaccess'

# Initialize S3 client using Boto3 (credentials from AWS CLI or environment variables)
s3_client = boto3.client('s3')

def check_s3_path_exists(bucket_name, path):
    """Check if an S3 path exists."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=path)
        if 'Contents' in response:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking path {path}: {e}")
        return False

def download_file_from_s3(bucket_name, s3_key, local_path):
    """Download file from S3."""
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        print(f"Downloaded {s3_key} to {local_path}")
    except NoCredentialsError:
        print("No AWS credentials found. Please ensure you are authenticated.")
    except PartialCredentialsError:
        print("Partial AWS credentials found. Please check your credentials.")
    except Exception as e:
        print(f"Error downloading file {s3_key}: {e}")

count = 0  # Counter for downloaded files

# Process each subject
for subject in subject_values:
    root = f"/scratch/alpine/alar6830/motor/{subject}/"
    RL_root = os.path.join(root, "RL")
    LR_root = os.path.join(root, "LR")

    # Ensure directories exist
    os.makedirs(RL_root, exist_ok=True)
    os.makedirs(LR_root, exist_ok=True)

    # Define S3 paths
    lr_s3_path = f"HCP_1200/{subject}/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz"
    rl_s3_path = f"HCP_1200/{subject}/MNINonLinear/Results/tfMRI_MOTOR_RL/tfMRI_MOTOR_RL.nii.gz"

    # Define local file paths
    lr_file = os.path.join(LR_root, "tfMRI_MOTOR_LR.nii.gz")
    rl_file = os.path.join(RL_root, "tfMRI_MOTOR_RL.nii.gz")

    # Download LR file if not exists
    if not os.path.exists(lr_file):
        if check_s3_path_exists(bucket_name, lr_s3_path):
            print(f"Downloading {lr_s3_path}...")
            download_file_from_s3(bucket_name, lr_s3_path, lr_file)
            count += 1
        else:
            print(f"Path NOT found: {lr_s3_path}")

    # Download RL file if not exists
    if not os.path.exists(rl_file):
        if check_s3_path_exists(bucket_name, rl_s3_path):
            print(f"Downloading {rl_s3_path}...")
            download_file_from_s3(bucket_name, rl_s3_path, rl_file)
            count += 1
        else:
            print(f"Path NOT found: {rl_s3_path}")

    print(f"Total files downloaded so far: {count}")

print(f"Final count of downloaded files: {count}")




