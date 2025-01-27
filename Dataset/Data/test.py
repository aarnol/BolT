import os
import numpy as np
import sys
import scipy.io
import torch
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prep'))
import fnirs_utils

# Define the path
# fnirs_data, MNI = fnirs_utils.load_fnirs(os.path.join(os.path.dirname(__file__), 'fNIRS'))

data = torch.load(os.path.join('Dataset', 'Data', 'motor_sphere.save'))
subjects = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101410', '101915', '102008']

# Filter the data for the specified subjects
filtered_data = [entry for entry in data if entry['pheno']['subjectId'] in subjects]

# Create the output directory if it doesn't exist
output_dir = os.path.join('Dataset', 'Data','motor', 'sphere_30')
os.makedirs(output_dir, exist_ok=True)

# Process each entry in the filtered data
for entry in filtered_data:
    rois = entry['roiTimeseries']
    pheno = entry['pheno']
    
    # Create a DataFrame from the ROI timeseries
    df = pd.DataFrame(rois)
    
    # Add the subject ID and task type to the DataFrame
    df['subjectId'] = pheno['subjectId']
    df['label'] = pheno['label']
    
    #label to str
   
    labels = {0: 'tongue', 1: 'right_hand', 2: 'right_foot', 3: 'left_hand', 4: 'left_foot'}
    label = labels[int(pheno['label'])]

    # Define the output file path
    output_file = os.path.join(output_dir, f"{pheno['subjectId']}_{label}.csv")
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

print("CSV files have been saved successfully.")