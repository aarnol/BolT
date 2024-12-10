import os
import nilearn as nil
import nilearn.datasets
import nilearn.image
from glob import glob
import pandas as pd
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker
from .prep_atlas import prep_atlas
from .fnirs_utils import load_fnirs, calc_MNI_average, process_fnirs
import numpy as np

datadir = "/scratch/alpine/alar6830/BoltROIs/"

def process_scan(scanImage_fileName, atlasImage):
    try:
        # Load the scan image and extract ROI time series
        scanImage = nil.image.load_img(scanImage_fileName)
        roiTimeseries = NiftiLabelsMasker(atlasImage).fit_transform(scanImage)
        
        # Extract subject ID, encoding, and n-back information based on file naming convention
        base_name = os.path.basename(scanImage_fileName)
        subjectId = base_name[:6]  # Adjust based on HCP filename structure
        enc = base_name[6]
        nback = base_name[8]

        # Return the processed data
        return {
            "roiTimeseries": roiTimeseries,
            "pheno": {
                "subjectId": subjectId,
                "encoding": enc,
                "nback": nback,
                "modality": "fMRI"
            }
        }
    except Exception as e:
        print(f"Error processing file {scanImage_fileName}: {e}")
        return None  # Return None if there's an error


def prep_abide(atlas, fnirs = False):
    bulkDataDir = "/scratch/alpine/alar6830/WM_nback_labels/"
    atlasImage = prep_atlas(atlas)

    if not os.path.exists(bulkDataDir):
        nil.datasets.fetch_abide_pcp(
            data_dir=bulkDataDir, pipeline="cpac", band_pass_filtering=False,
            global_signal_regression=False, derivatives="func_preproc", quality_checked=True
        )

    dataset = []

    # Load ABIDE phenotypic data
    temp = pd.read_csv(bulkDataDir + "/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv").to_numpy()
    phenoInfos = {str(row[2]): {"site": row[5], "age": row[9], "disease": row[7], "gender": row[10]} for row in temp}

    print("\n\nExtracting ROIs...\n\n")

    # Process each scan file in ABIDE data
    for scanImage_fileName in tqdm(glob(bulkDataDir + "/ABIDE_pcp/cpac/nofilt_noglobal/*"), ncols=60):
        if ".gz" in scanImage_fileName:
            try:
                scanImage = nil.image.load_img(scanImage_fileName)
                roiTimeseries = NiftiLabelsMasker(atlasImage).fit_transform(scanImage)
                subjectId = scanImage_fileName.split("_")[-3][2:]
                
                dataset.append({
                    "roiTimeseries": roiTimeseries,
                    "pheno": {
                        "subjectId": subjectId, **phenoInfos[subjectId]
                    }
                })
            except Exception as e:
                print(f"Error processing ABIDE file {scanImage_fileName}: {e}")

    torch.save(dataset, f"{datadir}/dataset_abide_{atlas}.save")


def prep_hcp(atlas, fnirs = False):
    # Define directory for HCP data
    bulkDataDir = "/scratch/alpine/alar6830/WM_nback_labels/"

    if(fnirs):
        fnirs_folder = ""
        data, digitization, timings = load_fnirs(fnirs_folder)
        MNI_coords = calc_MNI_average(digitization)
    else:
        MNI_coords = [[0, 0, 0],[100,100,100]]


    # Prepare the atlas image
    atlasImage = prep_atlas(atlas, datadir, MNI_coords)

    if not os.path.exists(bulkDataDir):
        raise Exception("Data does not exist")

    print("\n\nExtracting ROIs...\n\n")

    # Loop through HCP data files and process them in parallel
    scan_files = glob(bulkDataDir + "/*.nii.gz")

    # Process files in parallel with a progress bar
    with tqdm(total=len(scan_files), ncols=60) as pbar:
        dataset = []
        for result in Parallel(n_jobs=8)(
            delayed(process_scan)(file, atlasImage) for file in tqdm(scan_files, desc="Processing files")
        ):
            if result is not None:  # Only add successful results
                dataset.append(result)
            pbar.update(1)  # Update the progress bar
    #append fnirs data to dataset
    fnirs_data = process_fnirs(data, timings)
    for f in fnirs_data:
        dataset.append(f)
    # Save dataset
    torch.save(dataset, f"{datadir}/dataset_hcp_{atlas}.save")
