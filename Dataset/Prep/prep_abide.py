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

def process_scan(scanImage_fileName, MNI_coords, atlasImage =None,parcels = None, atlas = 'sphere', radius = 30, smooth_fwhm = None):
    try:
        # Load the scan image and extract ROI time series
        scanImage = nil.image.load_img(scanImage_fileName)
        # Apply smoothing
        if smooth_fwhm is not None:
            scanImage = nilearn.image.smooth_image(scanImage, fwhm = smooth_fwhm)
        roiTimeseries = []
        if(atlasImage == None):
            for coord in MNI_coords:
                MNI_values = calculate_average_bold(coord, scanImage.get_fdata(), scanImage.affine)
                roiTimeseries.append(MNI_values)
            roiTimeseries = np.array(roiTimeseries).T
        elif parcels != None:
            if(atlas == "AAL"):
                parcel_labels = [2001, 2002, 2101, 2102, 2111, 2112, 2201, 2202, 2211, 2212, 
                                2301, 2302, 2311, 2312, 2321, 2322, 2331, 2332, 2401, 2402, 
                                2501, 2502, 2601, 2602, 2611, 2612, 2701, 2702, 3001, 3002, 
                                4001, 4002, 4011, 4012, 4021, 4022, 4101, 4102, 4111, 4112, 
                                4201, 4202, 5001, 5002, 5011, 5012, 5021, 5022, 5101, 5102, 
                                5201, 5202, 5301, 5302, 5401, 5402, 6001, 6002, 6101, 6102, 
                                6201, 6202, 6211, 6212, 6221, 6222, 6301, 6302, 6401, 6402, 
                                7001, 7002, 7011, 7012, 7021, 7022, 7101, 7102, 8101, 8102, 
                                8111, 8112, 8121, 8122, 8201, 8202, 8211, 8212, 8301, 8302, 
                                9001, 9002, 9011, 9012, 9021, 9022, 9031, 9032, 9041, 9042, 
                                9051, 9052, 9061, 9062, 9071, 9072, 9081, 9082, 9100, 9110, 
                                9120, 9130, 9140, 9150, 9160, 9170]

                parcel_labels_dict = {label: index for index, label in enumerate(parcel_labels)}
            elif(atlas == "brodmann"):
                parcel_labels = my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
                parcel_labels_dict = {label:index for index, label in enumerate(parcel_labels)}
            else:
                raise NotImplementedError(f"{atlas} not supported") 


            masker = NiftiLabelsMasker(labels_img=atlasImage)
            parcel_signals = masker.fit_transform(scanImage).T
            
            for parcel in parcels:
                roiTimeseries.append(parcel_signals[parcel_labels_dict[int(parcel)]])
            roiTimeseries = np.array(roiTimeseries).T
            

                
            
        else:
            roiTimeseries = NiftiLabelsMasker(atlasImage).fit_transform(scanImage)
        
        # Extract subject ID, encoding, and n-back information based on file naming convention
        base_name = os.path.basename(scanImage_fileName)
        subjectId = base_name[:6]  # Adjust based on HCP filename structure
        enc = base_name[6]
        condition = base_name[7]
        nback = base_name[8]
        
        # Return the processed data
        
        return {
            "roiTimeseries": roiTimeseries,
            "pheno": {
                "subjectId": subjectId,
                "encoding": enc,
                "nback": nback,
                "condition" : condition,
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


def prep_hcp(atlas, name, fnirs = False, radius= 30, smooth_fwhm = None, unique_parcels = False):
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
            parcels.append(get_parcel_label(coord, atlasImage.get_fdata(), atlasImage.affine))
        if unique_parcels:
            unique_parcels = list(set(parcels))
            unique_parcel_indices = [parcels.index(parcel) for parcel in unique_parcels]
            print("Unique parcels:", unique_parcels)
            print("Indices of unique parcels:", unique_parcel_indices)
            parcels = np.array(parcels)[unique_parcel_indices].tolist()
            np.save(f"{datadir}/{atlas}_indices.txt", unique_parcel_indices)

    else:
        parcels = None


    if not os.path.exists(bulkDataDir):
        raise Exception("Data does not exist")

    print("\n\nExtracting ROIs...\n\n")

    # Loop through HCP data files and process them in parallel
    scan_files = glob(bulkDataDir + "/*.nii.gz")

    # Process files in parallel
    with tqdm(total=len(scan_files), ncols=60) as pbar:
        dataset = []
        for result in Parallel(n_jobs=256)(
            delayed(process_scan)(file, MNI_coords, atlasImage, parcels, atlas, radius = 30, smooth_fwhm = smooth_fwhm) for file in tqdm(scan_files, desc="Processing files")
        ):
            if result is not None:  # Only add successful results
                dataset.append(result)
            pbar.update(1)  # Update the progress bar
    
    # Save dataset
    torch.save(dataset, f"{datadir}/dataset_hcp_{atlas}_{name}.save")
