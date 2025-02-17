import os


import pandas as pd




#get data for downstream tasks
behavior_data = pd.read_csv("unrestricted_hcp_freesurfer.csv")

subject_values = behavior_data['Subject'].values

def check_file_name(file_name, downloaded_files):
  for file_name_path in downloaded_files:
    if file_name in file_name_path:
      return True
  return False
print(len(subject_values))

import subprocess

def check_s3_path_exists(bucket_name, path):
    aws_command = f"aws s3 ls s3://{bucket_name}/{path}"
    try:
        # Run the command and capture output
        result = subprocess.run(aws_command, shell=True, capture_output=True, text=True)
        
        # If the command returns 0, the path exists; otherwise, it doesn't
        if result.returncode == 0 and result.stdout.strip():
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking path {path}: {e}")
        return False



bucket_name = 'hcp-openaccess'

count = 0
for subject in subject_values:
    root = f"/scratch/alpine/alar6830/motor/{subject}/"
    RL_root = os.path.join(root, "RL")
    LR_root = os.path.join(root, "LR")
    
    
    if not os.path.exists(root):
        os.mkdir(root)
        os.mkdir(RL_root)
        os.mkdir(LR_root)
        
    s3_path = f"HCP_1200/{subject}/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz"
    
    file =os.path.join(LR_root, "tfMRI_MOTOR_LR.nii.gz" )
    if not os.path.exists(file):
        if check_s3_path_exists(bucket_name, s3_path):
            print("starting 1")
            path = f"s3://{bucket_name}/{s3_path}"
            exit()
            aws_command = f"aws s3 cp {path} {file}"
            os.system(aws_command)
            count+=1
            print("done with one")
        else:
            print(f"Path NOT found: {s3_path}")

    s3_path = f"HCP_1200/{subject}/MNINonLinear/Results/tfMRI_MOTOR_RL/tfMRI_MOTOR_RL.nii.gz"
    file =os.path.join(RL_root, "tfMRI_MOTOR_RL.nii.gz" )
    if not os.path.exists(file):
        if check_s3_path_exists(bucket_name, s3_path):
            
            path = f"s3://{bucket_name}/{s3_path}"
            aws_command = f"aws s3 cp {path} {file}"
            os.system(aws_command)
            count+=1
        else:
            print(f"Path NOT found: {s3_path}")

    
    print(count)

print(count)