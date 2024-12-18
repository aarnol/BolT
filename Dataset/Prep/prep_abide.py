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
from .fnirs_utils import load_fnirs, calculate_average_bold, get_parcel_label
import numpy as np

datadir = "/scratch/alpine/alar6830/BoltROIs/"

def process_scan(scanImage_fileName, MNI_coords, atlasImage =None,parcels = None, radius = 30):
    try:
        # Load the scan image and extract ROI time series
        scanImage = nil.image.load_img(scanImage_fileName)
        roiTimeseries = []
        if(atlasImage == None):
            for coord in MNI_coords:
                MNI_values = calculate_average_bold(coord, scanImage.get_fdata(), scanImage.affine)
                roiTimeseries.append(MNI_values)
            roiTimeseries = np.array(roiTimeseries).T
        elif parcels != None:
            pass
            # for parcel in parcels:
            #     masker = NiftiLabelsMasker(labels_img=atlasImage, labels=[parcel])
            #     parcel_timeseries = masker.fit_transform(scanImage)
            #     roiTimeseries.append(parcel_timeseries)
            # roiTimeseries = np.hstack(roiTimeseries)
        else:
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


def prep_hcp(atlas, name, fnirs = False):
    # Define directory for HCP data
    bulkDataDir = "/scratch/alpine/alar6830/WM_nback_labels/"

    if(fnirs):
        fnirs_folder = os.path.join(os.path.dirname(__file__), '..', 'Data','fNIRS')
        _, MNI_coords = load_fnirs(fnirs_folder)
        
    else:
        MNI_coords = None
    

    # Prepare the atlas image
    atlasImage = prep_atlas(atlas, datadir, MNI_coords)

    if(atlas!= "sphere" and fnirs):
        parcels = []
        for coord in MNI_coords:
            parcels.append(get_parcel_label(coord, atlasImage, atlasImage.affine))
    else:
        parcels = None


    if not os.path.exists(bulkDataDir):
        raise Exception("Data does not exist")

    print("\n\nExtracting ROIs...\n\n")

    # Loop through HCP data files and process them in parallel
    scan_files = glob(bulkDataDir + "/*.nii.gz")

    # Process files in parallel with a progress bar
    with tqdm(total=len(scan_files), ncols=60) as pbar:
        dataset = []
        for result in Parallel(n_jobs=256)(
            delayed(process_scan)(file, MNI_coords, atlasImage, parcels, radius = 30) for file in tqdm(scan_files, desc="Processing files")
        ):
            if result is not None:  # Only add successful results
                dataset.append(result)
            pbar.update(1)  # Update the progress bar
    
    # Save dataset
    torch.save(dataset, f"{datadir}/dataset_hcp_{atlas}_{name}.save")
